# CipherFace Integration

The React/Vite dashboard uses two FastAPI processes. The browser sends an image only to the trusted loopback client on port 8001. The client performs detection, landmark alignment, embedding extraction, fusion, normalization, CKKS encryption, result decryption, and the decision. The untrusted server on port 8002 accepts only encrypted templates and queries.

## Processes

- `client.local_api:app`: trusted device-local image pipeline and UI adapter. Bind only to `127.0.0.1:8001`.
- `server.api:app`: ciphertext-only verifier. Bind to a private interface or expose it through HTTPS in remote use.
- `frontend/`: Vite development server; `/api` is proxied to the local client service.

Do not launch `server/legacy_single_process_api.py` for privacy experiments. It is retained solely as the original demo implementation and accepts plaintext image uploads in a process that owns client decryption state.

## Local Startup

Install dependencies and initialize/migrate client keys as described in [README.md](README.md). Set the same locally generated `FACE_API_TOKEN` in the server and client process environments.

```bash
uvicorn server.api:app --host 127.0.0.1 --port 8002
```

In another terminal:

```bash
export FACE_API_TOKEN="<same local token>"
export FACE_SERVER_URL="http://127.0.0.1:8002"
uvicorn client.local_api:app --host 127.0.0.1 --port 8001
```

In a third terminal:

```bash
cd frontend
npm ci
npm run dev
```

## HTTP Boundaries

The server routes are `/api/health`, `/api/status`, `/api/templates`, `/api/enroll`, and `/api/verify`. Server enrollment and verification JSON bodies contain only a user identifier and base64 serialized CKKS ciphertext. The server returns encrypted similarity; accept/reject is decided by the local client after decryption.

The local client accepts image uploads at `/api/enroll` and `/api/verify`, but these are loopback client routes, not verifier routes. It does not save raw images or aligned crops. A separate `/api/verify-encrypted` route supports manual ciphertext protocol testing. LFW image browsing and evaluation-summary endpoints are served by the local client, not by the untrusted verifier.

The camera UI captures one still frame and closes the stream. It does not continuously upload video. The browser-to-local-client image request is trusted local processing; the client-to-server request contains ciphertext only.

## Research Artifacts

The official LFW DevTrain pairs provide validation; DevTest pairs provide the final test. The selected model and thresholds are in `results/final/selected_config_v3.json`. Machine-readable results and ROC plots are stored by model in `results/`. The baseline run is stored separately under `results/baseline/`.

For evaluation commands, measured results, security limitations, and key lifecycle, see [README.md](README.md) and [THREAT_MODEL.md](THREAT_MODEL.md).
