"""Trusted loopback service: process local images, send ciphertext to the server."""

import base64
import csv
import ipaddress
import json
import os
import re
import secrets
import sqlite3
import hashlib
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from stat import S_IMODE
from time import perf_counter, time
from urllib.error import HTTPError, URLError
from urllib.request import Request as UrlRequest, urlopen

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from client.pipeline import ClientPipeline
from client.verification import VerificationDecision
from common.config import CKKS, EMBEDDING, VERIFICATION
from common.security import PrivacyAudit, RequestRateLimiter, client_address, request_identifier
from database.metadata_manager import MetadataDatabase


app = FastAPI(
    title="CipherFace Local Client",
    version="2.0.0",
    description="Processes images locally and sends only CKKS ciphertext to the verifier.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def request_controls(request: Request, call_next):
    request_id = request_identifier(request)
    request.state.request_id = request_id
    host = request.client.host if request.client else None
    if _remote_client(host):
        response = JSONResponse({"detail": "Local client API accepts loopback connections only."}, status_code=403)
        response.headers["X-Request-ID"] = request_id
        return response
    origin = request.headers.get("origin")
    if not _origin_allowed(request.method, origin):
        response = JSONResponse({"detail": "Cross-origin mutation is not allowed."}, status_code=403)
        response.headers["X-Request-ID"] = request_id
        return response
    try:
        _RATE_LIMITER.check(
            client_address(request),
            request.method,
            12 if request.method == "POST" else 120,
        )
    except HTTPException as exc:
        response = JSONResponse({"detail": exc.detail}, status_code=exc.status_code)
    else:
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
    response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
    return response

_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_PIXELS = 20_000_000
_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_LFW_ROOT = Path("data/lfw/lfw-deepfunneled/lfw-deepfunneled")
_RESULTS = Path("results")
_CIPHERTEXT_LOG_DIR = Path("database/verification_ciphertexts")
_CIPHERTEXT_LOG_DIR.mkdir(parents=True, exist_ok=True)
_AUDIT = PrivacyAudit()
_RATE_LIMITER = RequestRateLimiter()
_ALLOWED_BROWSER_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173"}


def _origin_allowed(method: str, origin: str | None) -> bool:
    return method not in {"POST", "DELETE"} or origin is None or origin in _ALLOWED_BROWSER_ORIGINS


def _remote_client(host: str | None) -> bool:
    if not host:
        return True
    try:
        return not ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@lru_cache(maxsize=1)
def _client() -> ClientPipeline:
    return ClientPipeline()


def _server_url(path: str) -> str:
    base_url = os.environ.get("FACE_SERVER_URL", "http://127.0.0.1:8002").rstrip("/")
    return f"{base_url}{path}"


def _server_request(
    path: str,
    payload: dict | None = None,
    method: str = "GET",
    request_id: str | None = None,
) -> dict | list:
    token = os.environ.get("FACE_API_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="FACE_API_TOKEN is not configured on the local client.")
    if payload is not None:
        payload = {
            **payload,
            "issued_at": int(time()),
            "nonce": secrets.token_urlsafe(32),
        }
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = UrlRequest(
        _server_url(path),
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "X-Request-ID": request_id or secrets.token_hex(16),
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", "Verification server request failed.")
        except (ValueError, OSError):
            detail = "Verification server request failed."
        raise HTTPException(status_code=exc.code, detail=detail) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Ciphertext verification server is unavailable.") from exc


async def _load_image(upload: UploadFile) -> Image.Image:
    if not upload.content_type or not upload.content_type.startswith("image/"):
        await upload.close()
        raise HTTPException(status_code=415, detail="Upload an image file.")
    try:
        contents = bytearray(await upload.read(_MAX_IMAGE_BYTES + 1))
        if len(contents) > _MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the 10 MB limit.")
        with Image.open(BytesIO(contents)) as source:
            if source.width * source.height > _MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image dimensions exceed the 20 megapixel limit.")
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid face image.") from exc
    finally:
        if "contents" in locals():
            contents[:] = b"\x00" * len(contents)
        await upload.close()


def _safe_name(name: str) -> str:
    cleaned = name.strip()
    if not _NAME_PATTERN.fullmatch(cleaned):
        raise HTTPException(status_code=400, detail="Invalid user identifier.")
    return cleaned


def _inspect_locally(image: Image.Image) -> tuple[dict, bytes, dict[str, float]]:
    client = _client()
    timings: dict[str, float] = {}
    started = perf_counter()
    detection = client.detect_face(image)
    if detection is None:
        raise HTTPException(status_code=422, detail="No face detected. Choose or capture an image with a visible face.")
    timings["detection"] = (perf_counter() - started) * 1000

    facenet_embedding = arcface_embedding = fused = normalized = None
    try:
        started = perf_counter()
        facenet_embedding = client.facenet.extract(detection.aligned_face)
        timings["facenet"] = (perf_counter() - started) * 1000
        started = perf_counter()
        arcface_embedding = client.arcface.extract(detection.aligned_face)
        timings["arcface"] = (perf_counter() - started) * 1000
        timings["embedding"] = timings["facenet"] + timings["arcface"]

        started = perf_counter()
        fused = client.fusion.fuse(facenet_embedding, arcface_embedding)
        normalized = client.normalizer.normalize(fused)
        timings["fusion_normalization"] = (perf_counter() - started) * 1000

        started = perf_counter()
        encrypted = client.ckks.encrypt(normalized).serialize()
        timings["encryption"] = (perf_counter() - started) * 1000
        metadata = {
            "original": {"width": image.width, "height": image.height},
            "detection": {
                "confidence": round(detection.confidence, 5),
                "faces_detected": detection.face_count,
                "selected_face_policy": "largest bounding-box area; first MTCNN result breaks ties",
                "image_width": detection.image_width,
                "image_height": detection.image_height,
                "bbox": {
                    "x1": round(detection.bbox.x1, 2),
                    "y1": round(detection.bbox.y1, 2),
                    "x2": round(detection.bbox.x2, 2),
                    "y2": round(detection.bbox.y2, 2),
                },
                "landmarks": detection.landmarks.as_array().astype(float).tolist(),
            },
            "features": {
                "facenet": {"dimensions": int(facenet_embedding.shape[0]), "timing_ms": round(timings["facenet"], 2)},
                "arcface": {"dimensions": int(arcface_embedding.shape[0]), "timing_ms": round(timings["arcface"], 2)},
            },
            "fusion": {"input_dimensions": [512, 512], "output_dimensions": int(fused.shape[0])},
            "normalization": {
                "dimensions": int(normalized.shape[0]),
                "l2_norm": round(float(np.linalg.norm(normalized)), 6),
            },
            "encryption": {
                "scheme": "CKKS",
                "encrypted": True,
                "ciphertext_bytes": len(encrypted),
            },
        }
        return metadata, encrypted, timings
    finally:
        for vector in (facenet_embedding, arcface_embedding, fused, normalized):
            if isinstance(vector, np.ndarray):
                vector.fill(0)
        detection.aligned_face.fill(0)


async def _process_upload(upload: UploadFile) -> tuple[dict, bytes, dict[str, float]]:
    image = await _load_image(upload)
    try:
        return _inspect_locally(image)
    finally:
        image.close()


def _audit(
    request: Request,
    operation: str,
    stages: list[str],
    outcome: str,
    timings: dict[str, float] | None = None,
) -> list[dict]:
    request_id = request.state.request_id
    for stage in stages:
        _AUDIT.record(request_id, operation, stage, outcome)
    if timings:
        for stage, elapsed_ms in timings.items():
            _AUDIT.record(request_id, operation, stage, outcome, elapsed_ms)
    return _AUDIT.recent(30)



@app.post("/api/demo-login")
def demo_login(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(...)
) -> dict:
    credentials = {
        "client": {
            "username": "user",
            "password": "user123"
        },
        "server": {
            "username": "admin",
            "password": "admin123"
        }
    }

    expected = credentials.get(role)

    if not expected:
        raise HTTPException(status_code=401, detail="Invalid role.")

    if (
        not secrets.compare_digest(username, expected["username"])
        or not secrets.compare_digest(password, expected["password"])
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password."
        )

    return {
        "authenticated": True,
        "role": role,
        "mode": "development-demo-login"
    }

@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "role": "trusted-local-client"}


@app.get("/api/status")
def status() -> dict:
    server_status = _server_request("/api/status")
    _client()
    secret_path = Path(CKKS.secret_context_path).expanduser()
    secret_permissions = S_IMODE(secret_path.stat().st_mode) if secret_path.exists() else None
    return {
        **server_status,
        "models": {"mtcnn": True, "facenet": True, "arcface": True},
        "encryption": {
            **server_status.get("encryption", {}),
            "poly_modulus_degree": CKKS.poly_modulus_degree,
            "global_scale": f"2^{CKKS.global_scale_bits}",
        },
        "verification": {"threshold": VERIFICATION.similarity_threshold},
        "privacy_posture": {
            "client_models_loaded": True,
            "server_context_secret_free": server_status.get("encryption", {}).get("public_context_secret_free") is True,
            "secret_context_outside_project": secret_path.is_absolute() and not secret_path.is_relative_to(Path.cwd()),
            "secret_context_permissions_owner_only": secret_permissions == 0o600,
            "server_has_image_route": False,
            "server_has_plaintext_embedding_route": False,
            "server_authentication": server_status.get("security", {}).get("authentication") == "bearer-token-required",
            "replay_protection": server_status.get("security", {}).get("mutation_nonce_replay_protection") is True,
            "rate_limiting": server_status.get("security", {}).get("rate_limit", "not reported"),
            "plaintext_memory_cleanup": "best-effort array/buffer clearing; Python/ML allocator copies cannot be guaranteed erased",
        },
        "privacy_boundary": "image and secret key remain on this loopback client; server receives ciphertext only",
    }


@app.get("/api/templates")
def templates() -> list[dict]:
    result = _server_request("/api/templates")
    return result if isinstance(result, list) else []

@app.get("/api/verification-logs")
def verification_logs() -> list[dict]:
    metadata_db = MetadataDatabase()

    with metadata_db._connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT
                id,
                user_id,
                timestamp,
                decision,
                similarity,
                threshold,
                ciphertext_sha256,
                ciphertext_bytes,
                request_id
            FROM verification_logs
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]

@app.get("/api/verification-logs/{log_id}/ciphertext")
def get_verification_ciphertext(log_id: int) -> dict:
    if log_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid verification log ID.",
        )

    metadata_db = MetadataDatabase()

    with metadata_db._connect() as conn:
        conn.row_factory = sqlite3.Row
        log = conn.execute(
            """
            SELECT ciphertext_sha256, ciphertext_bytes
            FROM verification_logs
            WHERE id = ?
            """,
            (log_id,),
        ).fetchone()

    if log is None:
        raise HTTPException(
            status_code=404,
            detail="Verification log not found.",
        )

    ciphertext_path = _CIPHERTEXT_LOG_DIR / f"{log_id}.bin"

    if not ciphertext_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Ciphertext is unavailable for this verification log.",
        )

    ciphertext = ciphertext_path.read_bytes()
    actual_sha256 = hashlib.sha256(ciphertext).hexdigest()

    expected_sha256 = log["ciphertext_sha256"]
    expected_bytes = log["ciphertext_bytes"]

    if expected_sha256 and actual_sha256 != expected_sha256:
        raise HTTPException(
            status_code=409,
            detail="Ciphertext SHA-256 verification failed.",
        )

    if expected_bytes is not None and len(ciphertext) != expected_bytes:
        raise HTTPException(
            status_code=409,
            detail="Ciphertext size verification failed.",
        )

    return {
        "log_id": log_id,
        "ciphertext_base64": base64.b64encode(
            ciphertext
        ).decode("ascii"),
        "sha256": actual_sha256,
        "bytes": len(ciphertext),
        "integrity_ok": True,
    }

@app.get("/api/audit")
def audit_events() -> list[dict]:
    return _AUDIT.recent()


@app.delete("/api/templates")
def delete_template(request: Request, user_id: str = Form(...)) -> dict:
    safe_user_id = _safe_name(user_id)
    result = _server_request(
        "/api/templates",
        {"user_id": safe_user_id},
        method="DELETE",
        request_id=request.state.request_id,
    )
    _AUDIT.record(request.state.request_id, "template_deletion", "server_template_deleted", "success")
    return {"success": True, "user_id": safe_user_id, "request_id": request.state.request_id, "server_result": result}


@app.get("/api/research/summary")
def research_summary() -> dict:
    files = {
        "facenet": _RESULTS / "facenet/test_facenet_metrics.json",
        "arcface": _RESULTS / "arcface/test_arcface_metrics.json",
        "concatenation": _RESULTS / "fusion/test_concatenation_metrics.json",
        "fusion": _RESULTS / "fusion/test_selected_fusion_metrics.json",
        "ckks": _RESULTS / "encrypted/test_ckks_correctness_metrics.json",
    }
    try:
        metrics = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in files.items()}
        with (_RESULTS / "fusion/validation_grid_metrics_v3.csv").open(newline="", encoding="utf-8") as source:
            fusion_weights = list(csv.DictReader(source))
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="Recorded research results are unavailable.") from exc
    ckks = metrics["ckks"]
    return {
        "source": "results/ artifacts generated by the recorded LFW DevTest run",
        "pair_count": metrics["fusion"]["count"],
        "thresholds": {
            "facenet": metrics["facenet"]["threshold"],
            "arcface": metrics["arcface"]["threshold"],
            "concatenation": metrics["concatenation"]["threshold"],
            "selected_fusion": metrics["fusion"]["threshold"],
        },
        "models": {
            name: {
                key: metrics[name][key]
                for key in ("accuracy", "precision", "recall", "f1", "far", "frr", "eer_estimate", "auc", "threshold")
            }
            for name in ("facenet", "arcface", "concatenation", "fusion")
        },
        "fusion_selection": json.loads((_RESULTS / "final/selected_config_v3.json").read_text(encoding="utf-8"))["fusion"],
        "fusion_weight_grid": fusion_weights,
        "ckks": {
            "plaintext_metrics": ckks["plaintext_metrics"],
            "encrypted_metrics": ckks["encrypted_metrics"],
            "similarity_error": ckks["similarity_error"],
            "timing_seconds": ckks["timing_seconds"],
            "ciphertext_bytes": ckks["ciphertext_bytes"],
        },
    }


@app.post("/api/verify")
async def verify(
    request: Request,
    user_id: str = Form(...),
    image: UploadFile = File(...),
) -> dict:
    claimed_user = _safe_name(user_id)
    overall_started = perf_counter()

    metadata, encrypted_query, timings = await _process_upload(image)

    server_started = perf_counter()

    server_result = _server_request(
        "/api/verify",
        {
            "user_id": claimed_user,
            "encrypted_embedding": base64.b64encode(
                encrypted_query
            ).decode("ascii"),
        },
        method="POST",
        request_id=request.state.request_id,
    )

    timings["server_roundtrip"] = (
        perf_counter() - server_started
    ) * 1000

    try:
        encrypted_score = base64.b64decode(
            server_result["encrypted_similarity"],
            validate=True,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Server returned an invalid encrypted score.",
        ) from exc

    decrypt_started = perf_counter()

    similarity = _client().decrypt_similarity(encrypted_score)

    timings["decryption"] = (
        perf_counter() - decrypt_started
    ) * 1000

    timings["total"] = (
        perf_counter() - overall_started
    ) * 1000

    decision = VerificationDecision.decide(similarity)
    metadata_db = MetadataDatabase()

    metadata_db.add_verification_log(
        user_id=claimed_user,
        decision=decision["decision"],
        similarity=round(decision["similarity"], 6),
        threshold=VERIFICATION.similarity_threshold,
        ciphertext_sha256=server_result.get(
            "received_ciphertext_sha256"
        ),
        ciphertext_bytes=server_result.get(
            "received_ciphertext_bytes"
        ),
        request_id=request.state.request_id,
    )

    audit = _audit(
        request,
        "verification",
        [
            "image_processed_locally",
            "embedding_generated_locally",
            "embedding_encrypted",
            "ciphertext_transmitted",
            "ciphertext_only_matching",
            "encrypted_result_returned",
            "decision_completed_locally",
        ],
        "complete",
        timings,
    )

    return {
        "success": True,
        **metadata,
        "best_match": claimed_user,
        "similarity": round(decision["similarity"], 6),
        "threshold": VERIFICATION.similarity_threshold,
        "decision": decision["decision"],
        "verification_type": "claimed_identity_one_to_one",
        "server_input": "ENCRYPTED_QUERY_ONLY",
        "server_output": "ENCRYPTED_SIMILARITY_ONLY",
        "encrypted_embedding": base64.b64encode(
            encrypted_query
        ).decode("ascii"),
        "encrypted_embedding_encoding": "base64_ckks_ciphertext",
        "server_received_ciphertext_sha256": server_result.get(
            "received_ciphertext_sha256"
        ),
        "server_received_ciphertext_bytes": server_result.get(
            "received_ciphertext_bytes"
        ),
        "server_request_id": server_result.get("request_id"),
        "decryption": "local_client",
        "timing_ms": {
            key: round(value, 2)
            for key, value in timings.items()
        },
        "privacy_audit": audit,
        "request_id": request.state.request_id,
    }


@app.post("/api/enroll")
async def enroll(request: Request, name: str = Form(...), image: UploadFile = File(...)) -> dict:
    user_id = _safe_name(name)
    metadata, encrypted, timings = await _process_upload(image)
    server_started = perf_counter()
    server_result = _server_request(
        "/api/enroll",
        {"user_id": user_id, "encrypted_embedding": base64.b64encode(encrypted).decode("ascii")},
        method="POST",
        request_id=request.state.request_id,
    )
    timings["server_roundtrip"] = (perf_counter() - server_started) * 1000
    audit = _audit(
        request,
        "enrollment",
        ["image_processed_locally", "embedding_generated_locally", "template_encrypted", "ciphertext_template_transmitted", "template_stored_encrypted"],
        "complete",
        timings,
    )
    return {
        "success": True,
        "name": user_id,
        "status": "ENCRYPTED",
        "template_file": f"{user_id}.bin",
        "server_received": "encrypted_template_only",
        "local_processing": ["image_processed_locally", "embedding_generated_locally", "template_encrypted"],
        "encryption": metadata["encryption"],
        "server_result": server_result,
        "request_id": request.state.request_id,
        "timing_ms": {key: round(value, 2) for key, value in timings.items()},
        "privacy_audit": audit,
    }


@app.post("/api/verify-encrypted")
def verify_encrypted(
    request: Request,
    encrypted_embedding: str = Form(...),
    user_id: str = Form(...),
) -> dict:
    """
    Submit manually supplied Base64 CKKS ciphertext
    to the verification server and decrypt its encrypted score locally.
    """
    user_id = _safe_name(user_id)
    ciphertext_b64 = encrypted_embedding.strip()

    if not ciphertext_b64:
        raise HTTPException(
            status_code=400,
            detail="Please paste an encrypted ciphertext before verifying.",
        )

    try:
        ciphertext_bytes = base64.b64decode(
            ciphertext_b64,
            validate=True,
        )
    except (ValueError, base64.binascii.Error) as exc:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 ciphertext. Copy the complete encrypted embedding and try again.",
        ) from exc

    if not ciphertext_bytes:
        raise HTTPException(
            status_code=400,
            detail="The submitted ciphertext is empty.",
        )

    server_started = perf_counter()

    encrypted_result = _server_request(
        "/api/verify",
        {
            "user_id": user_id,
            "encrypted_embedding": base64.b64encode(
                ciphertext_bytes
            ).decode("ascii"),
        },
        method="POST",
        request_id=request.state.request_id,
    )

    server_roundtrip_ms = (perf_counter() - server_started) * 1000

    try:
        score_bytes = base64.b64decode(
            encrypted_result["encrypted_similarity"],
            validate=True,
        )
    except (KeyError, ValueError, base64.binascii.Error) as exc:
        raise HTTPException(
            status_code=502,
            detail="Server returned an invalid encrypted score.",
        ) from exc

    decrypt_started = perf_counter()
    similarity = _client().decrypt_similarity(score_bytes)
    decryption_ms = (perf_counter() - decrypt_started) * 1000

    decision = VerificationDecision.decide(similarity)
    metadata_db = MetadataDatabase()

    log_id = metadata_db.add_verification_log(
        user_id=user_id,
        decision=decision["decision"],
        similarity=round(decision["similarity"], 6),
        threshold=VERIFICATION.similarity_threshold,
        ciphertext_sha256=encrypted_result.get(
            "received_ciphertext_sha256"
        ),
        ciphertext_bytes=encrypted_result.get(
            "received_ciphertext_bytes"
        ),
        request_id=request.state.request_id,
    )

    ciphertext_path = _CIPHERTEXT_LOG_DIR / f"{log_id}.bin"
    temporary_path = _CIPHERTEXT_LOG_DIR / f"{log_id}.tmp"

    temporary_path.write_bytes(ciphertext_bytes)
    temporary_path.replace(ciphertext_path)

    return {
        "success": True,
        "best_match": user_id,
        "similarity": round(decision["similarity"], 6),
        "threshold": VERIFICATION.similarity_threshold,
        "decision": decision["decision"],
        "verification_type": "claimed_identity_one_to_one",
        "server_input": "ENCRYPTED_QUERY_ONLY",
        "server_output": "ENCRYPTED_SIMILARITY_ONLY",
        "server_received_ciphertext_sha256": encrypted_result.get(
            "received_ciphertext_sha256"
        ),
        "server_received_ciphertext_bytes": encrypted_result.get(
            "received_ciphertext_bytes"
        ),
        "server_request_id": encrypted_result.get("request_id"),
        "decryption": "local_client",
        "timing_ms": {
            "server_roundtrip": round(server_roundtrip_ms, 2),
            "decryption": round(decryption_ms, 2),
        },
        "request_id": request.state.request_id,
    }


@app.get("/api/lfw/summary")
def lfw_summary() -> dict:
    identities = sorted(path for path in _LFW_ROOT.iterdir() if path.is_dir())
    return {
        "dataset": "LFW deep-funneled",
        "images": sum(len(list(path.glob("*.jpg"))) for path in identities),
        "identities": len(identities),
    }


@app.get("/api/lfw/identities")
def lfw_identities(search: str = "") -> list[dict]:
    query = search.strip().lower()
    identities = []
    for path in sorted(path for path in _LFW_ROOT.iterdir() if path.is_dir()):
        if query and query not in path.name.lower():
            continue
        images = sorted(path.glob("*.jpg"))
        identities.append({"name": path.name, "image_count": len(images)})
    return identities[:100]


@app.get("/api/lfw/identity/{person}")
def lfw_identity(person: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_ -]+", person):
        raise HTTPException(status_code=400, detail="Invalid LFW identity.")
    directory = (_LFW_ROOT / person).resolve()
    if _LFW_ROOT.resolve() not in directory.parents:
        raise HTTPException(status_code=404, detail="LFW identity not found.")
    images = sorted(directory.glob("*.jpg"))
    if not images:
        raise HTTPException(status_code=404, detail="LFW identity not found.")
    return {"name": person, "image_count": len(images), "images": [image.name for image in images]}


@app.get("/api/lfw/image/{person}/{filename}")
def lfw_image(person: str, filename: str):
    from fastapi.responses import FileResponse

    if not re.fullmatch(r"[A-Za-z0-9_ -]+", person):
        raise HTTPException(status_code=400, detail="Invalid LFW identity.")
    directory = (_LFW_ROOT / person).resolve()
    path = (directory / filename).resolve()
    if _LFW_ROOT.resolve() not in directory.parents or path.parent != directory or path.suffix.lower() != ".jpg" or not path.is_file():
        raise HTTPException(status_code=404, detail="LFW image not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/evaluation/start")
def recorded_evaluation() -> dict:
    metrics_path = _RESULTS / "fusion/test_selected_fusion_ckks_metrics.json"
    scores_path = _RESULTS / "encrypted/test_similarity_comparison.csv"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with scores_path.open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=503, detail="Recorded final-test evaluation is unavailable.") from exc
    confusion = metrics["confusion_matrix"]
    return {
        "job_id": "recorded-lfw-devtest-ckks",
        "state": "complete",
        "processed": len(rows),
        "total": len(rows),
        "sample": None,
        "results": {
            "total_tests": len(rows),
            "genuine_tests": sum(row["label"] == "1" for row in rows),
            "impostor_tests": sum(row["label"] == "0" for row in rows),
            "true_accepts": confusion["true_accepts"],
            "true_rejects": confusion["true_rejects"],
            "false_accepts": confusion["false_accepts"],
            "false_rejects": confusion["false_rejects"],
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "far": round(metrics["far"] * 100, 2),
            "frr": round(metrics["frr"] * 100, 2),
            "threshold": metrics["threshold"],
            "average_similarity": round(float(np.mean([float(row["encrypted_score"]) for row in rows])), 4),
        },
    }


@app.get("/api/evaluation/{job_id}")
def evaluation_status(job_id: str) -> dict:
    if job_id != "recorded-lfw-devtest-ckks":
        raise HTTPException(status_code=404, detail="Recorded evaluation not found.")
    return recorded_evaluation()
