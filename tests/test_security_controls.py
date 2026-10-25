import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from fastapi import HTTPException, UploadFile
from starlette.requests import Request

from client import local_api
from common.security import PrivacyAudit, ReplayNonceCache, RequestRateLimiter
from server import privacy_api


class SecurityControlTests(unittest.TestCase):
    def test_replay_nonce_cache_rejects_stale_and_reused_requests(self):
        cache = ReplayNonceCache(ttl_seconds=30)
        cache.accept("N" * 32, issued_at=1000, now=1000)
        with self.assertRaises(HTTPException) as reused:
            cache.accept("N" * 32, issued_at=1000, now=1001)
        self.assertEqual(reused.exception.status_code, 409)
        with self.assertRaises(HTTPException) as stale:
            cache.accept("S" * 32, issued_at=900, now=1000)
        self.assertEqual(stale.exception.status_code, 401)

    def test_rate_limiter_returns_429_after_limit(self):
        limiter = RequestRateLimiter()
        limiter.check("client", "verify", limit=1)
        with self.assertRaises(HTTPException) as blocked:
            limiter.check("client", "verify", limit=1)
        self.assertEqual(blocked.exception.status_code, 429)

    def test_loopback_and_origin_guards_reject_remote_browser_mutations(self):
        self.assertFalse(local_api._remote_client("127.0.0.1"))
        self.assertFalse(local_api._remote_client("::1"))
        self.assertTrue(local_api._remote_client("192.0.2.8"))
        self.assertTrue(local_api._origin_allowed("POST", "http://localhost:5173"))
        self.assertFalse(local_api._origin_allowed("POST", "https://attacker.invalid"))
        self.assertTrue(local_api._origin_allowed("GET", "https://attacker.invalid"))

    def test_audit_event_schema_has_only_allowlisted_fields(self):
        audit = PrivacyAudit()
        audit.record("request-12345678", "verification", "ciphertext_sent", "success", 2.5)
        row = audit.recent()[0]
        self.assertEqual(
            set(row),
            {"timestamp", "request_id", "operation", "stage", "outcome", "latency_ms"},
        )
        self.assertNotIn("user_id", row)
        self.assertNotIn("embedding", row)
        self.assertNotIn("ciphertext", row)

    def test_browser_bundle_and_source_do_not_reference_secret_context(self):
        frontend = Path("frontend")
        sources = [frontend / "src/App.jsx", frontend / "src/api.js"]
        sources.extend((frontend / "dist/assets").glob("*.js"))
        combined = "\n".join(path.read_text(errors="ignore") for path in sources if path.is_file())
        self.assertNotIn("ckks_secret_context", combined)
        self.assertNotIn("FACE_CLIENT_SECRET_CONTEXT", combined)

    def test_server_has_no_plaintext_image_or_inspection_route(self):
        paths = privacy_api.app.openapi()["paths"]
        self.assertNotIn("/api/inspect", paths)
        self.assertNotIn("/api/enroll-image", paths)
        self.assertIn("/api/templates", paths)
        self.assertIn("delete", paths["/api/templates"])
        self.assertNotIn("/api/inspect", local_api.app.openapi()["paths"])
        for path in ("/api/enroll", "/api/verify"):
            content = paths[path]["post"]["requestBody"]["content"]
            self.assertIn("application/json", content)
            self.assertNotIn("multipart/form-data", content)

    def test_local_client_forwards_ciphertext_without_source_image_or_vectors(self):
        payloads = []
        encrypted_query = b"opaque-ciphertext"
        encrypted_score = b"opaque-encrypted-score"

        async def process_upload(_upload):
            return {"features": {"facenet": {"dimensions": 512}}}, encrypted_query, {"embedding": 3.0}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps({
                    "encrypted_similarity": base64.b64encode(encrypted_score).decode("ascii")
                }).encode("utf-8")

        def capture_request(remote_request, timeout):
            payloads.append(json.loads(remote_request.data.decode("utf-8")))
            return FakeResponse()

        local_client = Mock()
        local_client.decrypt_similarity.return_value = 0.9
        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": "/api/verify", "raw_path": b"/api/verify",
            "query_string": b"", "headers": [], "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8001),
        }
        request = Request(scope)
        request.state.request_id = "request-12345678"
        upload = UploadFile(filename="camera.jpg", file=tempfile.SpooledTemporaryFile())

        with (
            patch.object(local_api, "_process_upload", new=AsyncMock(side_effect=process_upload)),
            patch.object(local_api, "urlopen", side_effect=capture_request),
            patch.object(local_api, "_client", return_value=local_client),
            patch.dict(os.environ, {"FACE_API_TOKEN": "test-token"}),
        ):
            response = asyncio.run(local_api.verify(request, user_id="Known_User", image=upload))

        forwarded = payloads[0]
        self.assertEqual(set(forwarded), {"user_id", "encrypted_embedding", "issued_at", "nonce"})
        self.assertEqual(base64.b64decode(forwarded["encrypted_embedding"]), encrypted_query)
        self.assertNotIn(b"camera.jpg", repr(forwarded).encode())
        self.assertNotIn("facenet_embedding", forwarded)
        self.assertNotIn("arcface_embedding", forwarded)
        self.assertNotIn("plaintext_embedding", forwarded)
        self.assertEqual(response["decision"], "ACCEPT")
        self.assertEqual(response["decryption"], "local_client")
        upload.file.close()

    def test_mutable_biometric_arrays_are_cleared_after_encryption(self):
        aligned = np.ones((160, 160, 3), dtype=np.uint8)
        facenet = np.ones(512, dtype=np.float32)
        arcface = np.ones(512, dtype=np.float32)
        fused = np.ones(512, dtype=np.float32)
        normalized = np.ones(512, dtype=np.float32)
        detection = Mock(
            aligned_face=aligned,
            face_count=1,
            confidence=0.99,
            image_width=160,
            image_height=160,
            bbox=Mock(x1=0, y1=0, x2=1, y2=1),
            landmarks=Mock(as_array=lambda: np.zeros((5, 2))),
        )
        model_client = Mock(
            detect_face=Mock(return_value=detection),
            facenet=Mock(extract=Mock(return_value=facenet)),
            arcface=Mock(extract=Mock(return_value=arcface)),
            fusion=Mock(fuse=Mock(return_value=fused)),
            normalizer=Mock(normalize=Mock(return_value=normalized)),
            ckks=Mock(encrypt=Mock(return_value=Mock(serialize=Mock(return_value=b"cipher")))),
        )
        with patch.object(local_api, "_client", return_value=model_client):
            metadata, ciphertext, _ = local_api._inspect_locally(Mock(width=160, height=160))
        self.assertEqual(ciphertext, b"cipher")
        self.assertEqual(metadata["features"]["facenet"]["dimensions"], 512)
        self.assertFalse(aligned.any())
        self.assertFalse(facenet.any())
        self.assertFalse(arcface.any())
        self.assertFalse(fused.any())
        self.assertFalse(normalized.any())


if __name__ == "__main__":
    unittest.main()
