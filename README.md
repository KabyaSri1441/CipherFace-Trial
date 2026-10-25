# CipherFace Research Prototype

CipherFace is a face-verification research prototype using MTCNN, FaceNet, ArcFace, CKKS encrypted similarity, FastAPI, and a React/Vite dashboard. The application now separates the trusted local client from the ciphertext-only verification server. It is research-grade, not production biometric infrastructure.

## Architecture

```text
Trusted client (local machine)                 Untrusted verification server
Camera or selected image                       Encrypted template store
  -> MTCNN and five-point alignment              <- encrypted enrollment template
  -> FaceNet + ArcFace                           <- encrypted query
  -> validation-selected weighted fusion          homomorphic dot product
  -> L2 normalization                           -> encrypted similarity
  -> CKKS encrypt                                -> return ciphertext
  <- encrypted similarity
  -> decrypt locally
  -> claimed-identity threshold decision
```

The local FastAPI client service handles image files on the user device at `127.0.0.1:8001`. The remote service in `server/api.py` receives JSON containing a user identifier and base64 serialized CKKS ciphertext only. It has no image-upload or inspection route. `server/legacy_single_process_api.py` preserves the old single-process demo for reference; **do not run it for privacy experiments**.

## Face Processing

`client/detector.py` uses `facenet-pytorch` MTCNN and its five landmarks. It estimates a similarity transform to a canonical five-point template and produces a 160x160 RGB `uint8` face. Multiple-face policy is explicit: select the detected face with the largest bounding-box area; if areas tie, use the first MTCNN result. If no face or usable landmarks are returned, processing stops with an error. The aligned crop is not returned by production API responses.

FaceNet is `facenet-pytorch` InceptionResnetV1 pretrained on VGGFace2 and emits 512-D normalized features. ArcFace is InsightFace `buffalo_l/w600k_r50.onnx` and emits 512-D normalized features. The selected production fusion is a normalized weighted sum:

```text
normalize(0.4 * FaceNet + 0.6 * ArcFace)
```

It is 512-D. Concatenation remains available through `FeatureFusion.concatenate` and is evaluated separately; it is not assumed to be superior.

## CKKS and Key Lifecycle

TenSEAL CKKS parameters: polynomial modulus degree 8192, coefficient modulus `[60, 40, 40, 60]`, and global scale `2^40`. The client encrypts only the final normalized embedding. Since embeddings are unit-normalized, the server dot product approximates cosine similarity. The client decrypts the encrypted scalar and applies the decision threshold.

The secret context lives outside the project at `~/.local/share/cipherface/client/ckks_secret_context.bin` with owner-only permissions. The server receives only `server/config/ckks_public_context.bin`; startup refuses a context that has a secret key. Existing legacy templates are 1024-D concatenated vectors and are preserved in `database/encrypted_templates/`; the selected 512-D pipeline uses a separate v2 database. Existing identities need to be enrolled again for the v2 method. No legacy template or key artifact is silently deleted.

For a fresh client keypair, run `python -m scripts.initialize_client_keys`. To migrate the existing local keypair and retain access to existing legacy artifacts, run `python -m scripts.relocate_ckks_keys` once. Do not copy the secret context to a server host or commit it. `.gitignore` excludes local secrets and biometric database files.

## Install and Run

Use Python 3.11 and Node.js with npm. From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m scripts.initialize_client_keys
```

For an existing installation with the generated key in the old location, run `python -m scripts.relocate_ckks_keys` instead of generating a new keypair. Existing encrypted templates only work with their original keypair.

Configure the same bearer token in the separate server and client terminals. Generate it locally; do not put it in source control:

```bash
export FACE_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
```

Run the untrusted server (only public context and v2 ciphertext storage):

```bash
uvicorn server.api:app --host 127.0.0.1 --port 8002
```

Run the trusted local client in a second terminal with the same `FACE_API_TOKEN`:

```bash
export FACE_API_TOKEN="<same locally generated token>"
export FACE_SERVER_URL="http://127.0.0.1:8002"
uvicorn client.local_api:app --host 127.0.0.1 --port 8001
```

Run the dashboard in a third terminal:

```bash
cd frontend
npm ci
npm run dev
```

Open the Vite URL, normally `http://localhost:5173`. The browser's camera is accessed only after explicit permission. The UI captures a single still, stops the stream, and sends the still to the trusted loopback client; image-upload fallback remains available. The local client returns a result/status summary, not image pixels or plaintext embeddings.

For a remote server set `FACE_SERVER_URL` to its HTTPS URL and deploy behind TLS. The loopback client API is intentionally bound to `127.0.0.1`; exposing that image-processing endpoint to a network would change the privacy boundary.

## API

The default `server.api:app` routes are `GET /api/health`, authenticated `GET /api/status`, authenticated `GET /api/templates`, authenticated `POST /api/enroll`, and authenticated `POST /api/verify`. Enrollment/verification bodies are JSON with an identifier and base64 CKKS ciphertext. The server returns an encrypted similarity; it does not decide accept/reject.

The trusted `client.local_api:app` accepts local multipart image uploads at `POST /api/enroll` and `POST /api/verify`, plus an explicit ciphertext transfer endpoint at `POST /api/verify-encrypted`. It also serves local LFW browsing and recorded evaluation results. Enrollment and verification images are processed in memory; the local API does not persist them. The debug script `scripts/test_mtcnn.py` writes visualizations under `data/debug/` and is not used by production API paths.

Server bearer authentication is a prototype shared-token control. The client API is loopback-bound and CORS-restricted to the local Vite origin, but CORS is not authentication. Rate limiting, user accounts, audit policy, and deployment-grade identity authorization are not implemented.

## Evaluation Protocol

The supplied LFW official DevTrain pair files are used as validation data and DevTest pair files as final test data. The identity sets are disjoint. Fixed source and scored lists are stored under `results/protocol/`; the one validation pair with a face MTCNN did not detect is recorded in `validation_detection_failures.csv` and excluded identically from every validation model. All 1,000 DevTest pairs were scoreable. Existing `data/lfw_scores.txt` was preserved.

Run the protocol preparation and experiments:

```bash
python -m scripts.prepare_lfw_protocol
python -m scripts.evaluate_models --split validation
python -m scripts.select_validation_config
python -m scripts.evaluate_models --split test
python -m scripts.report_test_models
python -m scripts.evaluate_ckks
```

Each result writer uses exclusive creation and refuses to overwrite existing artifacts. Run a new experiment under a new result path/version if regenerating. Validation selects fusion by highest AUC, then lowest EER estimate; thresholds are the ROC operating point minimizing absolute FAR-FRR difference. The final test uses those frozen thresholds. Test ROC/AUC/EER are descriptive; they do not alter the selected threshold.

### Measured Results

These values are from actual executions on the supplied LFW official split with landmark alignment. They are benchmark results, not deployment guarantees.

| Model | Accuracy | FAR | FRR | EER estimate | AUC | Decision threshold |
|---|---:|---:|---:|---:|---:|---:|
| FaceNet | 93.5% | 7.2% | 5.8% | 5.9% | 0.97158 | 0.29839 |
| ArcFace | 94.3% | 5.8% | 5.6% | 4.7% | 0.97078 | 0.09777 |
| Concatenation | 94.4% | 5.4% | 5.8% | 4.3% | 0.97226 | 0.18562 |
| Weighted fusion (`alpha=0.4`) | 94.5% | 5.2% | 5.8% | 6.1% | 0.97364 | 0.14208 |

The selected weighted fusion has the highest test accuracy/AUC of these four on this run, but its test EER estimate is worse than concatenation. Selection was made on validation AUC, not test data. Results and full confusion matrices/ROC files are under `results/facenet/`, `results/arcface/`, `results/fusion/`, and `results/final/`.

On the same 1,000 test pairs, plaintext and CKKS weighted-fusion decisions matched exactly:

| Privacy configuration | Accuracy | FAR | FRR | EER estimate | AUC |
|---|---:|---:|---:|---:|---:|
| Plaintext | 94.5% | 5.2% | 5.8% | 6.1% | 0.97364 |
| CKKS | 94.5% | 5.2% | 5.8% | 6.1% | 0.97364 |

Mean absolute similarity error was `4.48e-7` (maximum `8.62e-7`); mean relative error was `1.59e-5`. On this CPU run, per-pair encryption averaged 9.47 ms, homomorphic deserialization/dot/serialization 14.00 ms, and decryption 1.13 ms. Mean serialized query/template/result ciphertext sizes were about 331,390 / 331,361 / 235,072 bytes. End-to-end processing of all pairs took 215.43 seconds, including approximately 188.37 seconds of detection and feature extraction. See `results/encrypted/test_ckks_correctness_metrics.json` for details and timing definitions.

The baseline before the upgrades is separately recorded in `results/baseline/baseline.json`: the old evaluator produced 19/20 correct at its hard-coded 0.55 threshold. The old API threshold was 0.247. These historical values are not mixed with the new official-protocol results.

## Tests and Limitations

Run `python -m unittest discover -s tests -v`. Tests cover LFW pair loading and split separation, no/single/multiple face handling, alignment output, fusion behavior, ciphertext size/auth checks, and the server key/privacy boundary. A real one-image detector check, full validation/test model runs, CKKS-vs-plaintext run, and frontend production build were also executed during this upgrade.

The server is assumed honest-but-curious, not malicious. CKKS does not provide liveness, protection against compromised clients, protection from traffic/access-pattern analysis, full template unlinkability, or protection against malicious chosen queries. User IDs and timing/request metadata remain plaintext. There is no rate limiter. The current static bearer token is shared by client and server and should be replaced by proper service identity/secret management for deployment. Camera support depends on browser permission and secure-context rules; remote use requires HTTPS. LFW is a benchmark with demographic and capture limitations.
