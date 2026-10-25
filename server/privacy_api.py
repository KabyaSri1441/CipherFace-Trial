"""Ciphertext-only API intended to run as the untrusted verification server."""

import base64
import binascii
import hmac
import hashlib
import os
import re
from functools import lru_cache
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from common.config import CKKS
from common.security import (
    PrivacyAudit,
    ReplayNonceCache,
    RequestRateLimiter,
    client_address,
    request_identifier,
)
from server.pipeline import ServerService


app = FastAPI(
    title="CipherFace Ciphertext API",
    version="2.0.0",
    description="Receives encrypted templates and queries only.",
)
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_MAX_CIPHERTEXT_BYTES = 10 * 1024 * 1024
_AUDIT = PrivacyAudit()
_RATE_LIMITER = RequestRateLimiter()
_REPLAY_CACHE = ReplayNonceCache(ttl_seconds=180)


class EncryptedRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    encrypted_embedding: str = Field(min_length=1, max_length=14_000_000)
    issued_at: int
    nonce: str = Field(min_length=24, max_length=128)


class DeleteRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    issued_at: int
    nonce: str = Field(min_length=24, max_length=128)


@app.middleware("http")
async def security_headers_and_request_id(request: Request, call_next):
    request_id = request_identifier(request)
    request.state.request_id = request_id
    response = await call_next(request)
    if response.status_code >= 400:
        route = request.scope.get("route")
        operation = getattr(route, "name", request.method.lower())
        _AUDIT.record(request_id, str(operation), "request_rejected", "failure")
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cache-Control"] = "no-store"
    return response


def _authorize(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    limit = 20 if request.method in {"POST", "DELETE"} else 120
    _RATE_LIMITER.check(client_address(request), request.method, limit)
    expected = os.environ.get("FACE_API_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="Server authentication is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Bearer authentication is required.")
    supplied = authorization.removeprefix("Bearer ")
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid bearer token.")


def _decode_ciphertext(encoded: str) -> bytes:
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Ciphertext must be valid Base64.") from exc
    if not decoded or len(decoded) > _MAX_CIPHERTEXT_BYTES:
        raise HTTPException(status_code=413, detail="Ciphertext is empty or exceeds the size limit.")
    return decoded


def _validate_user_id(user_id: str) -> str:
    if not _NAME_PATTERN.fullmatch(user_id):
        raise HTTPException(status_code=400, detail="Invalid user identifier.")
    return user_id


def _accept_request_nonce(nonce: str, issued_at: int) -> None:
    _REPLAY_CACHE.accept(nonce, issued_at)


@lru_cache(maxsize=1)
def _service() -> ServerService:
    context_path = Path(os.environ.get("FACE_SERVER_PUBLIC_CONTEXT", CKKS.context_path))
    try:
        public_context = context_path.read_bytes()
    except OSError as exc:
        raise RuntimeError("Public CKKS context is unavailable.") from exc
    service = ServerService(public_context)
    if service.ctx.context.has_secret_key():
        raise RuntimeError("Refusing to start: server context contains a secret key.")
    return service


@app.get("/api/health")
def health(request: Request) -> dict:
    _RATE_LIMITER.check(client_address(request), "health", 120)
    return {"status": "ok", "mode": "ciphertext-only"}


@app.get("/api/status")
def status(_: None = Depends(_authorize)) -> dict:
    service = _service()
    return {
        "system": "ONLINE",
        "encryption": {
            "scheme": "CKKS",
            "server_mode": "CIPHERTEXT_ONLY",
            "public_context_secret_free": not service.ctx.context.has_secret_key(),
            "secret_key_loaded": service.ctx.context.has_secret_key(),
        },
        "security": {
            "authentication": "bearer-token-required",
            "mutation_nonce_replay_protection": True,
            "rate_limit": "20 mutations/minute and 120 reads/minute per process/IP",
        },
        "database": {"status": True, "enrolled_users": len(service.list_users())},
    }


@app.get("/api/templates")
def templates(_: None = Depends(_authorize)) -> list[dict]:
    service = _service()
    metadata = service.metadata_db.list_users()
    return [
        {
            "name": row[1],
            "template_file": Path(row[2]).name,
            "status": row[3],
            "created_at": row[4],
        }
        for row in metadata
    ]


@app.post("/api/enroll")
def enroll(
    payload: EncryptedRequest,
    request: Request,
    _: None = Depends(_authorize),
) -> dict:
    _accept_request_nonce(payload.nonce, payload.issued_at)
    user_id = _validate_user_id(payload.user_id)
    service = _service()
    if service.user_exists(user_id):
        raise HTTPException(status_code=409, detail="Identity is already enrolled.")
    encrypted_template = _decode_ciphertext(payload.encrypted_embedding)
    try:
        service.ctx.load_encrypted_vector(encrypted_template)
        service.enroll(user_id, encrypted_template)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid encrypted template.") from exc
    _AUDIT.record(request.state.request_id, "enrollment", "ciphertext_template_stored", "complete")
    return {"success": True, "user_id": user_id, "status": "ENCRYPTED", "request_id": request.state.request_id}


@app.post("/api/verify")
def verify(
    payload: EncryptedRequest,
    request: Request,
    _: None = Depends(_authorize),
) -> dict:
    _accept_request_nonce(payload.nonce, payload.issued_at)
    user_id = _validate_user_id(payload.user_id)
    encrypted_query = _decode_ciphertext(payload.encrypted_embedding)
    service = _service()
    if not service.user_exists(user_id):
        raise HTTPException(status_code=404, detail="No enrolled template for this identity.")
    try:
        encrypted_similarity = service.verify(user_id, encrypted_query)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid encrypted query.") from exc
    _AUDIT.record(request.state.request_id, "verification", "ciphertext_match_complete", "success")
    return {
        "success": True,
        "user_id": user_id,
        "encrypted_similarity": base64.b64encode(encrypted_similarity).decode("ascii"),
        # Server-observed metadata lets the local UI verify which serialized
        # ciphertext arrived without exposing plaintext or any secret key.
        "received_ciphertext_sha256": hashlib.sha256(encrypted_query).hexdigest(),
        "received_ciphertext_bytes": len(encrypted_query),
        "request_id": request.state.request_id,
    }


@app.delete("/api/templates")
def delete_template(
    payload: DeleteRequest,
    request: Request,
    _: None = Depends(_authorize),
) -> dict:
    _accept_request_nonce(payload.nonce, payload.issued_at)
    user_id = _validate_user_id(payload.user_id)
    service = _service()
    if not service.delete_user(user_id):
        raise HTTPException(status_code=404, detail="Identity was not found.")
    _AUDIT.record(request.state.request_id, "template_deletion", "encrypted_template_deleted", "success")
    return {"success": True, "user_id": user_id, "request_id": request.state.request_id}


@app.get("/api/audit")
def audit_events(_: None = Depends(_authorize)) -> list[dict]:
    return _AUDIT.recent()
