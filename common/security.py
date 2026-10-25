"""Small in-process security controls and allowlisted privacy audit events."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from time import time
from uuid import uuid4

from fastapi import HTTPException, Request


@dataclass(frozen=True)
class AuditEvent:
    timestamp: float
    request_id: str
    operation: str
    stage: str
    outcome: str
    latency_ms: float | None = None


class PrivacyAudit:
    def __init__(self, max_events: int = 500):
        self._events: deque[AuditEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def record(
        self,
        request_id: str,
        operation: str,
        stage: str,
        outcome: str,
        latency_ms: float | None = None,
    ) -> None:
        if outcome not in {"success", "failure", "complete"}:
            raise ValueError("Unsupported audit outcome.")
        event = AuditEvent(time(), request_id, operation, stage, outcome, latency_ms)
        with self._lock:
            self._events.append(event)

    def recent(self, limit: int = 100) -> list[dict]:
        with self._lock:
            events = list(self._events)[-max(0, min(limit, 100)):]
        return [
            {
                "timestamp": event.timestamp,
                "request_id": event.request_id,
                "operation": event.operation,
                "stage": event.stage,
                "outcome": event.outcome,
                "latency_ms": event.latency_ms,
            }
            for event in events
        ]


class RequestRateLimiter:
    """Per-process sliding-window limiter; use a shared store for multi-worker deploys."""

    def __init__(self):
        self._requests: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, client_key: str, bucket: str, limit: int, window_seconds: int = 60) -> None:
        now = time()
        key = (client_key, bucket)
        with self._lock:
            requests = self._requests[key]
            while requests and requests[0] <= now - window_seconds:
                requests.popleft()
            if len(requests) >= limit:
                raise HTTPException(status_code=429, detail="Request rate limit exceeded.")
            requests.append(now)


class ReplayNonceCache:
    """Short-lived replay detection for authenticated mutating protocol requests."""

    def __init__(self, ttl_seconds: int = 180, max_entries: int = 100_000):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._nonces: dict[str, float] = {}
        self._lock = Lock()

    def accept(self, nonce: str, issued_at: int, now: float | None = None) -> None:
        current = time() if now is None else now
        if not nonce or len(nonce) < 24 or len(nonce) > 128:
            raise HTTPException(status_code=400, detail="Invalid request nonce.")
        if abs(current - issued_at) > self.ttl_seconds:
            raise HTTPException(status_code=401, detail="Request timestamp is outside the accepted window.")
        with self._lock:
            expired = [key for key, expires_at in self._nonces.items() if expires_at <= current]
            for key in expired:
                del self._nonces[key]
            if nonce in self._nonces:
                raise HTTPException(status_code=409, detail="Request nonce has already been used.")
            if len(self._nonces) >= self.max_entries:
                raise HTTPException(status_code=503, detail="Replay protection capacity reached.")
            self._nonces[nonce] = current + self.ttl_seconds


def request_identifier(request: Request) -> str:
    supplied = request.headers.get("x-request-id", "")
    if 8 <= len(supplied) <= 64 and all(char.isalnum() or char in "-_" for char in supplied):
        return supplied
    return uuid4().hex


def client_address(request: Request) -> str:
    return request.client.host if request.client else "unknown"
