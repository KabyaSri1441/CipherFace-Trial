import os
import base64
import secrets
import stat
import unittest
from pathlib import Path
from time import time
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

from server import privacy_api


class PrivacyBoundaryTests(unittest.TestCase):
    @staticmethod
    def _request():
        return Request({
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
            "method": "GET", "scheme": "http", "path": "/api/status", "raw_path": b"/api/status",
            "query_string": b"", "headers": [], "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8002),
        })

    def test_server_openapi_has_no_image_or_inspection_route(self):
        schema = privacy_api.app.openapi()

        self.assertNotIn("/api/inspect", schema["paths"])
        self.assertNotIn("/api/evaluation/start", schema["paths"])
        for path in ("/api/enroll", "/api/verify"):
            request_body = schema["paths"][path]["post"]["requestBody"]
            self.assertIn("application/json", request_body["content"])
            self.assertNotIn("multipart/form-data", request_body["content"])

    def test_ciphertext_validation_rejects_invalid_and_oversized_payloads(self):
        with self.assertRaises(HTTPException) as invalid:
            privacy_api._decode_ciphertext("not base64!")
        self.assertEqual(invalid.exception.status_code, 400)

        oversized = base64.b64encode(b"\x00" * (10 * 1024 * 1024 + 1)).decode("ascii")
        with self.assertRaises(HTTPException) as too_large:
            privacy_api._decode_ciphertext(oversized)
        self.assertEqual(too_large.exception.status_code, 413)

    def test_server_requires_configured_bearer_token(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as missing_config:
                privacy_api._authorize(self._request(), "Bearer any")
        self.assertEqual(missing_config.exception.status_code, 503)

        with patch.dict(os.environ, {"FACE_API_TOKEN": "test-token"}, clear=True):
            with self.assertRaises(HTTPException) as invalid_token:
                privacy_api._authorize(self._request(), "Bearer wrong-token")
            privacy_api._authorize(self._request(), "Bearer test-token")
        self.assertEqual(invalid_token.exception.status_code, 401)

    def test_server_service_loads_public_only_context_and_v2_storage(self):
        privacy_api._service.cache_clear()
        service = privacy_api._service()

        self.assertFalse(service.ctx.context.has_secret_key())
        self.assertEqual(service.encrypted_db.database_dir, Path("database/encrypted_templates_v2"))
        self.assertEqual(service.metadata_db.db_path, Path("database/metadata_v2.db"))

    def test_server_refuses_a_secret_bearing_context(self):
        from common.config import CKKS

        privacy_api._service.cache_clear()
        with patch.dict(os.environ, {"FACE_SERVER_PUBLIC_CONTEXT": CKKS.secret_context_path}):
            with self.assertRaisesRegex(RuntimeError, "secret key"):
                privacy_api._service()
        privacy_api._service.cache_clear()

    def test_server_rejects_replayed_mutation_nonce(self):
        nonce = secrets.token_urlsafe(32)
        issued_at = int(time())
        privacy_api._accept_request_nonce(nonce, issued_at)
        with self.assertRaises(HTTPException) as replay:
            privacy_api._accept_request_nonce(nonce, issued_at)
        self.assertEqual(replay.exception.status_code, 409)

    def test_secret_context_is_outside_workspace(self):
        from common.config import CKKS

        self.assertTrue(Path(CKKS.secret_context_path).is_absolute())
        self.assertFalse(Path(CKKS.secret_context_path).is_relative_to(Path.cwd()))
        self.assertEqual(stat.S_IMODE(Path(CKKS.secret_context_path).stat().st_mode), 0o600)


if __name__ == "__main__":
    unittest.main()
