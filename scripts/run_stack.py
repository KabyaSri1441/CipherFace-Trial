"""Start the ciphertext server (8002) and trusted local client (8001) with one shared token.

The bearer token is only ever shared between the two backend processes (via the
environment). It is never sent to or bundled into the browser: the browser talks
to the loopback client, and the client authenticates to the server.

Usage:  python -m scripts.run_stack            # backends only
        python -m scripts.run_stack --frontend # also start the Vite dev server
"""

import argparse
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen
import threading

from common.config import CKKS

TOKEN_PATH = Path.home() / ".local/share/cipherface/server_api_token"
SERVER_PORT = 8002
CLIENT_PORT = 8001


def load_token() -> str:
    token = os.environ.get("FACE_API_TOKEN")
    if token:
        return token
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if TOKEN_PATH.exists():
        return TOKEN_PATH.read_text().strip()
    token = secrets.token_urlsafe(32)
    fd = os.open(TOKEN_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as target:
        target.write(token)
    return token


def ensure_keys() -> None:
    """Make sure the client keypair and the secret-free server context both exist."""
    from scripts.initialize_client_keys import main as init_keys

    init_keys()
    if not Path(CKKS.context_path).exists():
        raise SystemExit("Public CKKS context is missing after key initialization.")


def wait_healthy(port: int, name: str, proc: subprocess.Popen, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"{name} exited early (code {proc.returncode}).")
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise SystemExit(f"{name} did not become healthy on port {port}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", action="store_true", help="also run the Vite dev server")
    args = parser.parse_args()

    ensure_keys()
    env = {
        **os.environ,
        "FACE_API_TOKEN": load_token(),
        "FACE_SERVER_URL": f"http://127.0.0.1:{SERVER_PORT}",
        "PYTHONPATH": os.getcwd(),
    }
    uvicorn = [sys.executable, "-m", "uvicorn", "--host", "127.0.0.1"]
    procs = [subprocess.Popen(uvicorn + ["server.api:app", "--port", str(SERVER_PORT)], env=env)]
    try:
        wait_healthy(SERVER_PORT, "ciphertext server", procs[0])
        procs.append(subprocess.Popen(uvicorn + ["client.local_api:app", "--port", str(CLIENT_PORT)], env=env))
        wait_healthy(CLIENT_PORT, "local client", procs[1])
        if args.frontend:
            procs.append(subprocess.Popen(["npm.cmd", "run", "dev"], cwd="frontend"))
        print(f"Server http://127.0.0.1:{SERVER_PORT} | Client http://127.0.0.1:{CLIENT_PORT}"
              + (" | UI http://localhost:5173" if args.frontend else ""), flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()


if __name__ == "__main__":
    main()
