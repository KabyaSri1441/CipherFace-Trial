# Threat Model

## Goal

Protect face images, plaintext face embeddings, plaintext similarity scores, and the CKKS secret key from an honest-but-curious verification server during enrollment and claimed-identity verification.

## Assumption

The verification server follows the request protocol but may inspect every byte stored or transmitted to it. It is not assumed to be maliciously modifying ciphertexts in this prototype. Clients and their host machines are trusted for biometric processing and decryption.

## Trusted Components

- The local client process running `client.local_api:app` on loopback.
- The client-side detector, FaceNet/ArcFace models, fusion, normalizer, and CKKS secret context.
- The client operating system and local key-file permissions.

## Untrusted Components

- The verification server running `server.api:app`.
- Its ciphertext template database and metadata database.
- Network links between the local client service and a remotely deployed server. Use HTTPS for remote deployment.

## Data Flow

1. The browser obtains one uploaded image or captures one still after camera permission.
2. The image is sent only to the trusted local loopback client service.
3. The local client detects/alines the face, computes embeddings, fuses/normalizes them, and encrypts the result.
4. Enrollment sends an identifier and encrypted template to the server. Verification sends a claimed identifier and encrypted query.
5. The server computes an encrypted dot product and returns encrypted similarity.
6. The local client decrypts the score and makes the accept/reject decision.

## Server-Visible Information

- Encrypted enrolled template ciphertexts.
- Encrypted query ciphertexts.
- Encrypted similarity ciphertexts.
- Plaintext account/user identifier, request timing, endpoint, ciphertext size, request frequency, and access patterns.
- Public CKKS parameters/context.

The user identifier may directly reveal a person's name in the demo. Use opaque random IDs where names are not required.

## Server Must Not Receive

- Raw face photographs or camera frames.
- Plaintext FaceNet or ArcFace embeddings.
- Plaintext fused embeddings.
- Plaintext similarity scores or accept/reject decisions.
- The CKKS secret key or secret-bearing context.

The standard server app's OpenAPI accepts ciphertext JSON and has no image upload, inspection, camera, or decryption endpoint. The secret context defaults to `~/.local/share/cipherface/client/ckks_secret_context.bin`; server loading verifies `has_secret_key()` is false. Do not deploy the legacy `server/legacy_single_process_api.py`.

## Key Management

The client creates and retains the secret-bearing context under a user-local directory with owner-only permissions. The server receives only `server/config/ckks_public_context.bin`. The old key artifacts were migrated without changing the keypair so existing ciphertexts remain decryptable; the old secret backup is also outside the project. Do not place the client key directory in a server image, backup, or shared volume. Rotate/re-enroll templates if keys are rotated.

## Metadata and Retention

The server stores the user ID, template filename, encrypted template, status, and creation timestamp. The API process does not persist uploaded raw images or aligned crops. The debug-only MTCNN script writes image artifacts under `data/debug/`; keep that out of production workflows. Plaintext model embeddings and decrypted scores exist in local client memory; evaluation score reports are local research artifacts.

## Explicit Limitations

CKKS encryption alone does not provide:

- Liveness or presentation-attack detection.
- Protection from a compromised client, browser, or client operating system.
- Protection from malicious clients submitting chosen/adaptive queries.
- Protection from server denial of service, malicious result manipulation, or rollback.
- Traffic-analysis or access-pattern privacy.
- Complete unlinkability of templates across databases.
- Identity/account security, rate limiting, or a full authorization lifecycle.
- Correctness proofs for server computation or malicious-secure MPC.

The shared bearer token is a research-prototype service credential, not end-user authentication. The client API must remain loopback-bound; making it network-accessible would expose raw images to that network endpoint. Remote client/server deployment requires HTTPS and operational secret management. No production-security claim is made.
