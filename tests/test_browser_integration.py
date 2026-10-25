"""Browser-path regression tests: local client route -> real ciphertext server -> local decision.

Face models are stubbed (heavy weights); CKKS encryption, the server app, bearer auth, nonce
replay protection, template storage and homomorphic similarity are all real.
"""

import base64
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import tenseal as ts
from fastapi.testclient import TestClient
from PIL import Image

from client import local_api
from server import privacy_api

TOKEN = "integration-test-token"


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class BrowserIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.secret_ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.secret_ctx.generate_galois_keys()
        self.secret_ctx.global_scale = 2 ** 40
        public_path = root / "public.bin"
        public_path.write_bytes(self.secret_ctx.serialize(save_secret_key=False))

        env = {
            "FACE_API_TOKEN": TOKEN,
            "FACE_SERVER_URL": "http://ciphertext-server.test",
            "FACE_SERVER_PUBLIC_CONTEXT": str(public_path),
            "FACE_TEMPLATE_DATABASE": str(root / "templates"),
            "FACE_METADATA_DATABASE": str(root / "meta.db"),
        }
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        privacy_api._service.cache_clear()
        self.addCleanup(privacy_api._service.cache_clear)

        self.server = TestClient(privacy_api.app)
        self.client = TestClient(local_api.app)
        self.forwarded: list[dict] = []
        self.embedding = {}

        def fake_urlopen(request, timeout=0):
            path = request.full_url.removeprefix("http://ciphertext-server.test")
            body = json.loads(request.data) if request.data else None
            if body is not None:
                self.forwarded.append(body)
            response = self.server.request(
                request.get_method(), path, content=request.data,
                headers={k: v for k, v in request.header_items()},
            )
            if response.status_code >= 400:
                raise HTTPError(request.full_url, response.status_code, "err", {}, BytesIO(response.content))
            return _FakeResponse(response.content)

        def fake_inspect(image):
            vector = self.embedding["vector"]
            encrypted = ts.ckks_vector(self.secret_ctx, vector.tolist()).serialize()
            metadata = {"encryption": {"scheme": "CKKS", "encrypted": True, "ciphertext_bytes": len(encrypted)}}
            return metadata, encrypted, {}

        class _StubPipeline:
            def decrypt_similarity(inner, data):
                return ts.ckks_vector_from(self.secret_ctx, data).decrypt()[0]

        for p in (
            patch("client.local_api.urlopen", fake_urlopen),
            patch("client.local_api._inspect_locally", fake_inspect),
            patch("client.local_api._client", lambda: _StubPipeline()),
        ):
            p.start()
            self.addCleanup(p.stop)

    @staticmethod
    def _unit(seed: int) -> np.ndarray:
        vector = np.random.default_rng(seed).normal(size=512)
        return vector / np.linalg.norm(vector)

    @staticmethod
    def _png() -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (32, 32), "white").save(buffer, "PNG")
        return buffer.getvalue()

    def _enroll(self, name, seed):
        self.embedding["vector"] = self._unit(seed)
        return self.client.post(
            "/api/enroll", data={"name": name}, files={"image": ("a.png", self._png(), "image/png")}
        )

    def _verify(self, name, seed):
        self.embedding["vector"] = self._unit(seed)
        return self.client.post(
            "/api/verify", data={"user_id": name}, files={"image": ("a.png", self._png(), "image/png")}
        )

    def test_enroll_route_succeeds_and_server_gets_ciphertext_only(self):
        response = self._enroll("alice", 1)

        self.assertEqual(response.status_code, 200, response.text)  # was a 500 KeyError ('client_stages')
        self.assertEqual(response.json()["server_received"], "encrypted_template_only")
        self.assertEqual(set(self.forwarded[0]), {"user_id", "encrypted_embedding", "issued_at", "nonce"})
        service = privacy_api._service()
        self.assertFalse(service.ctx.context.has_secret_key())
        stored = service.encrypted_db.get_template("alice")
        plain = self.embedding["vector"].astype(np.float64).tobytes()
        self.assertNotIn(plain[:64], stored)
        self.assertEqual(self.client.get("/api/templates").json()[0]["name"], "alice")

    def test_encrypted_verification_accepts_same_and_rejects_different(self):
        self._enroll("alice", 1)

        same = self._verify("alice", 1).json()
        different = self._verify("alice", 2).json()

        self.assertEqual(same["decision"], "ACCEPT")
        self.assertAlmostEqual(same["similarity"], 1.0, places=3)
        self.assertEqual(different["decision"], "REJECT")
        self.assertEqual(same["server_output"], "ENCRYPTED_SIMILARITY_ONLY")
        self.assertNotIn("encrypted_similarity", same)

    def test_status_route_reports_online_through_authenticated_server(self):
        self._enroll("alice", 1)
        with patch("client.local_api.Path.exists", return_value=False):
            status = self.client.get("/api/status").json()
        self.assertEqual(status["system"], "ONLINE")
        self.assertEqual(status["database"]["enrolled_users"], 1)
        self.assertTrue(status["privacy_posture"]["server_authentication"])

    def test_server_without_configured_token_is_503_and_without_header_401(self):
        """The reported symptom: pointing the browser at the server process (no browser token)."""
        response = self.server.get("/api/status")
        self.assertEqual(response.status_code, 401)
        with patch.dict(os.environ, {"FACE_API_TOKEN": ""}):
            self.assertEqual(self.server.get("/api/status").status_code, 503)

    def test_frontend_bundle_source_never_references_the_server_token(self):
        for path in Path("frontend/src").glob("*.js*"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("FACE_API_TOKEN", text, path)
            self.assertNotIn("Authorization", text, path)


if __name__ == "__main__":
    unittest.main()
