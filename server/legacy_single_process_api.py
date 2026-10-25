"""Thin HTTP adapter for the existing privacy-preserving face pipeline.

This module deliberately does not implement biometric or encryption logic. The
local demo process hosts both sides, but server computation and client-side
 decryption remain separate operations in the request flow.
"""

from functools import lru_cache
from pathlib import Path
from time import perf_counter
import re
import base64
from io import BytesIO
import threading
import uuid
import os

_PROJECT_TORCH_HOME = Path(".runtime-cache/torch").resolve()
os.environ.setdefault("TORCH_HOME", str(_PROJECT_TORCH_HOME))
_PROJECT_HOME = Path(".runtime-cache/home").resolve()
if not os.access(Path.home(), os.W_OK):
    _PROJECT_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(_PROJECT_HOME)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
import numpy as np

from client.pipeline import ClientPipeline
from client.verification import VerificationDecision
from common.config import CKKS, EMBEDDING, VERIFICATION
from database.metadata_manager import MetadataDatabase
from server.pipeline import ServerService
from server.matcher import EncryptedMatcher


app = FastAPI(
    title="CipherFace API",
    version="1.0.0",
    description="Privacy-preserving face verification over encrypted templates.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,79}$")
_LFW_ROOT = Path("data/lfw/lfw-deepfunneled/lfw-deepfunneled")
_evaluation_jobs: dict[str, dict] = {}
_evaluation_lock = threading.Lock()


def _lfw_person(person: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_ -]+", person):
        raise HTTPException(status_code=400, detail="Invalid LFW identity.")
    path = (_LFW_ROOT / person).resolve()
    if _LFW_ROOT.resolve() not in path.parents:
        raise HTTPException(status_code=404, detail="LFW identity not found.")
    return path


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
    path = _lfw_person(person)
    images = sorted(path.glob("*.jpg"))
    if not images:
        raise HTTPException(status_code=404, detail="LFW identity not found.")
    return {"name": person, "image_count": len(images), "images": [image.name for image in images]}


@app.get("/api/lfw/image/{person}/{filename}")
def lfw_image(person: str, filename: str) -> FileResponse:
    path = _lfw_person(person) / filename
    if path.parent != _lfw_person(person) or path.suffix.lower() != ".jpg" or not path.is_file():
        raise HTTPException(status_code=404, detail="LFW image not found.")
    return FileResponse(path, media_type="image/jpeg")


@lru_cache(maxsize=1)
def _runtime() -> tuple[ClientPipeline, ServerService, MetadataDatabase]:
    _PROJECT_TORCH_HOME.mkdir(parents=True, exist_ok=True)
    client = ClientPipeline()
    public_context = client.public_context_bytes()
    return client, ServerService(public_context), MetadataDatabase()


def _load_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(upload.file).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Upload a valid face image.") from exc


def _safe_name(name: str) -> str:
    cleaned = name.strip()
    if not _NAME_PATTERN.fullmatch(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Name must be 1-80 characters and contain only letters, numbers, spaces, ., _, or -.",
        )
    return cleaned


def _metadata_row(row: tuple) -> dict:
    _, person_name, template_file, status, created_at = row
    return {
        "name": person_name,
        "template_file": Path(template_file).name,
        "status": status,
        "created_at": created_at,
    }


def _decrypt_on_client(client: ClientPipeline, encrypted_similarity: bytes) -> float:
    """Local-demo client step: decrypt using the secret-bearing client context."""
    return float(client.decrypt_similarity(encrypted_similarity))


def _image_data_url(image: np.ndarray) -> str:
    output = BytesIO()
    Image.fromarray(image.astype(np.uint8), mode="RGB").save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _evaluation_cases(metadata_db: MetadataDatabase) -> list[dict]:
    people = []
    for person_dir in sorted(_LFW_ROOT.iterdir()):
        images = sorted(person_dir.glob("*.jpg"))
        if images:
            people.append((person_dir.name, images))
    if len(people) < 2:
        raise HTTPException(status_code=409, detail="At least two LFW identities are required for evaluation.")

    genuine_pool = [(name, images[0], images[1], name) for name, images in people if len(images) >= 2]
    impostor_pool = [(name, images[0], people[(index + 1) % len(people)][0], people[(index + 1) % len(people)][1][0]) for index, (name, images) in enumerate(people)]
    if len(genuine_pool) < 250:
        raise HTTPException(status_code=409, detail="The local LFW dataset does not provide 250 same-identity pairs.")
    cases = [{"kind": "genuine", "probe_name": name, "probe": str(probe), "template_name": target, "template": str(template)} for name, probe, template, target in genuine_pool[:250]]
    cases.extend({"kind": "impostor", "probe_name": name, "probe": str(probe), "template_name": target, "template": str(template)} for name, probe, target, template in impostor_pool[:250])
    return cases


def _run_evaluation(job_id: str, cases: list[dict]) -> None:
    client, server, _ = _runtime()
    results = []
    for index, case in enumerate(cases, start=1):
        try:
            image = Image.open(case["probe"]).convert("RGB")
            encrypted_probe = client.image_to_encrypted_embedding(image)
            if encrypted_probe is None:
                continue
            encrypted_template = client.image_to_encrypted_embedding(Image.open(case["template"]).convert("RGB"))
            if encrypted_template is None:
                continue
            probe_vector = server.ctx.load_encrypted_vector(encrypted_probe)
            template_vector = server.ctx.load_encrypted_vector(encrypted_template)
            encrypted_similarity = EncryptedMatcher.compute_encrypted_similarity(probe_vector, template_vector).serialize()
            similarity = _decrypt_on_client(client, encrypted_similarity)
            accepted = similarity >= VERIFICATION.similarity_threshold
            results.append({"kind": case["kind"], "accepted": accepted, "similarity": similarity})
            with _evaluation_lock:
                _evaluation_jobs[job_id]["sample"] = {
                    "probe_name": case["probe_name"], "target_name": case["template_name"],
                    "kind": case["kind"], "similarity": round(float(similarity), 4),
                    "decision": "ACCEPT" if accepted else "REJECT",
                    "probe_path": case["probe"],
                    "template_path": case["template"],
                }
        finally:
            with _evaluation_lock:
                _evaluation_jobs[job_id]["processed"] = index

    genuine = [row for row in results if row["kind"] == "genuine"]
    impostor = [row for row in results if row["kind"] == "impostor"]
    true_accepts = sum(row["accepted"] for row in genuine)
    true_rejects = sum(not row["accepted"] for row in impostor)
    false_rejects = len(genuine) - true_accepts
    false_accepts = len(impostor) - true_rejects
    total = len(results)
    with _evaluation_lock:
        _evaluation_jobs[job_id].update({
            "state": "complete", "results": {
                "total_tests": total, "genuine_tests": len(genuine), "impostor_tests": len(impostor),
                "true_accepts": true_accepts, "true_rejects": true_rejects,
                "false_accepts": false_accepts, "false_rejects": false_rejects,
                "accuracy": round((true_accepts + true_rejects) / total * 100, 2) if total else 0,
                "far": round(false_accepts / len(impostor) * 100, 2) if impostor else 0,
                "frr": round(false_rejects / len(genuine) * 100, 2) if genuine else 0,
                "average_similarity": round(sum(row["similarity"] for row in results) / total, 4) if total else 0,
                "threshold": VERIFICATION.similarity_threshold,
            },
        })


def _run_evaluation_safe(job_id: str, cases: list[dict]) -> None:
    try:
        _run_evaluation(job_id, cases)
    except Exception as exc:
        with _evaluation_lock:
            _evaluation_jobs[job_id].update({"state": "failed", "error": str(exc)})


@app.post("/api/evaluation/start")
def start_evaluation() -> dict:
    metadata_db = MetadataDatabase()
    with _evaluation_lock:
        running = next((job_id for job_id, job in _evaluation_jobs.items() if job["state"] == "running"), None)
    if running:
        return {"job_id": running, "state": "running", "total": 500}
    cases = _evaluation_cases(metadata_db)
    job_id = uuid.uuid4().hex
    with _evaluation_lock:
        _evaluation_jobs[job_id] = {"state": "running", "processed": 0, "total": len(cases), "sample": None, "results": None}
    threading.Thread(target=_run_evaluation_safe, args=(job_id, cases), daemon=True).start()
    return {"job_id": job_id, "state": "running", "total": len(cases)}


@app.get("/api/evaluation/{job_id}")
def evaluation_status(job_id: str) -> dict:
    with _evaluation_lock:
        job = _evaluation_jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Evaluation job not found.")
        response = {key: value for key, value in job.items() if key not in {"probe_path", "template_path"}}
        if response.get("sample"):
            response["sample"] = {
                key: value for key, value in response["sample"].items()
                if key not in {"probe_path", "template_path"}
            }
        return response


@app.get("/api/evaluation/{job_id}/image/{side}")
def evaluation_image(job_id: str, side: str) -> FileResponse:
    with _evaluation_lock:
        job = _evaluation_jobs.get(job_id)
        sample = job.get("sample") if job else None
    if not sample or side not in {"probe", "template"}:
        raise HTTPException(status_code=404, detail="Evaluation sample image not available.")
    path = Path(sample[f"{side}_path"]).resolve()
    if _LFW_ROOT.resolve() not in path.parents or path.suffix.lower() != ".jpg":
        raise HTTPException(status_code=404, detail="Evaluation image not available.")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/inspect")
def inspect(image: UploadFile = File(...)) -> dict:
    """Run the client stages and return safe visualization metadata only."""
    client, _, _ = _runtime()
    raw_image = _load_image(image)
    started = perf_counter()
    detection = client.detect_face(raw_image)
    if detection is None:
        raise HTTPException(status_code=422, detail="No face detected in the selected image.")

    facenet_started = perf_counter()
    facenet_embedding = client.facenet.extract(detection.aligned_face)
    facenet_ms = (perf_counter() - facenet_started) * 1000
    arcface_started = perf_counter()
    arcface_embedding = client.arcface.extract(detection.aligned_face)
    arcface_ms = (perf_counter() - arcface_started) * 1000
    fused = client.fusion.fuse(facenet_embedding, arcface_embedding)
    normalized = client.normalizer.normalize(fused)
    encrypted = client.ckks.encrypt(normalized).serialize()

    return {
        "success": True,
        "original": {"width": raw_image.width, "height": raw_image.height},
        "detection": {
            "confidence": round(detection.confidence, 5),
            "image_width": detection.image_width,
            "image_height": detection.image_height,
            "bbox": {
                "x1": round(detection.bbox.x1, 2), "y1": round(detection.bbox.y1, 2),
                "x2": round(detection.bbox.x2, 2), "y2": round(detection.bbox.y2, 2),
            },
            "landmarks": [[round(float(x), 2), round(float(y), 2)] for x, y in detection.landmarks.as_array()],
        },
        "aligned_face": {"width": 160, "height": 160, "image": _image_data_url(detection.aligned_face)},
        "features": {
            "facenet": {"dimensions": int(facenet_embedding.shape[0]), "timing_ms": round(facenet_ms, 1)},
            "arcface": {"dimensions": int(arcface_embedding.shape[0]), "timing_ms": round(arcface_ms, 1)},
        },
        "fusion": {"input_dimensions": [512, 512], "output_dimensions": int(fused.shape[0])},
        "normalization": {"dimensions": int(normalized.shape[0]), "l2_norm": round(float(np.linalg.norm(normalized)), 5)},
        "encryption": {"scheme": CKKS.__class__.__name__.replace("Config", ""), "encrypted": bool(encrypted), "payload_exposed": False},
        "encrypted_embedding": base64.b64encode(encrypted).decode("ascii"),
        "timing_ms": {"client_pipeline": round((perf_counter() - started) * 1000, 1)},
    }


@app.post("/api/verify-encrypted")
def verify_encrypted(encrypted_embedding: str = Form(...)) -> dict:
    """Verify a serialized ciphertext received from the separate Client demo."""
    try:
        encrypted_probe = base64.b64decode(encrypted_embedding.strip(), validate=True)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Encrypted embedding must be valid Base64 text.") from exc
    if not encrypted_probe:
        raise HTTPException(status_code=400, detail="Encrypted embedding cannot be empty.")

    client, server, metadata_db = _runtime()
    rows = metadata_db.list_users()
    if not rows:
        raise HTTPException(status_code=409, detail="No encrypted templates are enrolled.")
    scores = []
    for row in rows:
        encrypted_similarity = server.verify(row[1], encrypted_probe)
        scores.append((row[1], _decrypt_on_client(client, encrypted_similarity)))
    best_match, similarity = max(scores, key=lambda item: item[1])
    decision = VerificationDecision.decide(similarity)
    return {
        "success": True,
        "best_match": best_match,
        "similarity": round(decision["similarity"], 4),
        "threshold": VERIFICATION.similarity_threshold,
        "decision": decision["decision"],
        "server_input": "ENCRYPTED_EMBEDDING_ONLY",
    }


@app.get("/api/status")
def status() -> dict:
    metadata_db = MetadataDatabase()
    return {
        "system": "ONLINE",
        "models": {
            "mtcnn": True,
            "facenet": True,
            "arcface": True,
        },
        "encryption": {
            "scheme": "CKKS",
            "poly_modulus_degree": CKKS.poly_modulus_degree,
            "global_scale": f"2^{CKKS.global_scale_bits}",
            "server_mode": "CIPHERTEXT_ONLY",
        },
        "database": {
            "status": Path("database/metadata.db").exists(),
            "enrolled_users": metadata_db.count_users(),
        },
        "verification": {
            "threshold": VERIFICATION.similarity_threshold,
        },
    }


@app.get("/api/templates")
def templates() -> list[dict]:
    metadata_db = MetadataDatabase()
    return [_metadata_row(row) for row in metadata_db.list_users()]


@app.post("/api/enroll")
def enroll(name: str = Form(...), image: UploadFile = File(...)) -> dict:
    person_name = _safe_name(name)
    client, server, _ = _runtime()

    started = perf_counter()
    encrypted_embedding = client.image_to_encrypted_embedding(_load_image(image))
    if encrypted_embedding is None:
        raise HTTPException(status_code=422, detail="No face detected in the enrollment image.")

    server.enroll(person_name, encrypted_embedding)
    template_file = f"{person_name}.bin"
    return {
        "success": True,
        "name": person_name,
        "template_file": template_file,
        "status": "ENCRYPTED",
        "timing_ms": {"client_encryption": round((perf_counter() - started) * 1000, 1)},
    }


@app.post("/api/verify")
def verify(image: UploadFile = File(...)) -> dict:
    client, server, metadata_db = _runtime()
    started = perf_counter()
    encrypted_probe = client.image_to_encrypted_embedding(_load_image(image))
    if encrypted_probe is None:
        raise HTTPException(status_code=422, detail="No face detected in the probe image.")
    client_encryption_ms = (perf_counter() - started) * 1000

    rows = metadata_db.list_users()
    if not rows:
        raise HTTPException(status_code=409, detail="No encrypted templates are enrolled.")

    matching_started = perf_counter()
    encrypted_results = [
        (row[1], server.verify(row[1], encrypted_probe))
        for row in rows
    ]
    server_matching_ms = (perf_counter() - matching_started) * 1000

    decrypt_started = perf_counter()
    scores = [
        (name, _decrypt_on_client(client, encrypted_similarity))
        for name, encrypted_similarity in encrypted_results
    ]
    best_match, similarity = max(scores, key=lambda item: item[1])
    decision = VerificationDecision.decide(similarity)
    client_decryption_ms = (perf_counter() - decrypt_started) * 1000

    return {
        "success": True,
        "best_match": best_match,
        "similarity": round(decision["similarity"], 4),
        "threshold": VERIFICATION.similarity_threshold,
        "decision": decision["decision"],
        "timing_ms": {
            "client_encryption": round(client_encryption_ms, 1),
            "server_matching": round(server_matching_ms, 1),
            "client_decryption": round(client_decryption_ms, 1),
        },
    }


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
