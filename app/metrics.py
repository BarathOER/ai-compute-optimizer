"""In-process metrics: request counts, latency, and estimated token cost.

Cost is estimated, not billed. Tokens are approximated from character length
(~4 chars/token, a standard rough heuristic) and priced per model via config.
The key business metric is *savings*: for every request we compute what it
would have cost to always hit the remote model, and subtract what it actually
cost. Cache hits and local routes both contribute savings.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string (~4 characters per token)."""
    return max(1, len(text) // 4)


@dataclass
class MetricsSnapshot:
    """A point-in-time, serializable view of the collected metrics."""

    total_requests: int
    cache_hits: int
    cache_misses: int
    local_routes: int
    remote_routes: int
    hit_rate: float
    avg_latency_ms: float
    avg_hit_latency_ms: float
    avg_miss_latency_ms: float
    total_cost_usd: float
    total_baseline_cost_usd: float
    total_savings_usd: float


@dataclass
class Metrics:
    """Thread-safe accumulator for gateway metrics."""

    remote_cost_per_1k: float
    local_cost_per_1k: float = 0.0

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    local_routes: int = 0
    remote_routes: int = 0
    _latencies_ms: list[float] = field(default_factory=list)
    _hit_latencies_ms: list[float] = field(default_factory=list)
    _miss_latencies_ms: list[float] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_baseline_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cost_for(self, tokens: int, *, remote: bool) -> float:
        """Return the estimated USD cost of ``tokens`` on the given backend."""
        rate = self.remote_cost_per_1k if remote else self.local_cost_per_1k
        return (tokens / 1000.0) * rate

    def record(
        self,
        *,
        cache_hit: bool,
        route: str,
        latency_ms: float,
        tokens: int,
    ) -> tuple[float, float]:
        """Record one request and return ``(actual_cost, savings)`` for it.

        ``savings`` is the remote baseline cost minus the actual cost, i.e. what
        was avoided by serving from cache or routing to the local model.
        """
        baseline_cost = self.cost_for(tokens, remote=True)
        if cache_hit:
            actual_cost = 0.0
        else:
            actual_cost = self.cost_for(tokens, remote=(route == "remote"))
        savings = baseline_cost - actual_cost

        with self._lock:
            self.total_requests += 1
            self._latencies_ms.append(latency_ms)
            if cache_hit:
                self.cache_hits += 1
                self._hit_latencies_ms.append(latency_ms)
            else:
                self.cache_misses += 1
                self._miss_latencies_ms.append(latency_ms)
                if route == "remote":
                    self.remote_routes += 1
                else:
                    self.local_routes += 1
            self.total_cost_usd += actual_cost
            self.total_baseline_cost_usd += baseline_cost

        return actual_cost, savings

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def snapshot(self) -> MetricsSnapshot:
        """Return an immutable, serializable snapshot of current metrics."""
        with self._lock:
            total = self.total_requests
            hit_rate = self.cache_hits / total if total else 0.0
            return MetricsSnapshot(
                total_requests=total,
                cache_hits=self.cache_hits,
                cache_misses=self.cache_misses,
                local_routes=self.local_routes,
                remote_routes=self.remote_routes,
                hit_rate=hit_rate,
                avg_latency_ms=self._avg(self._latencies_ms),
                avg_hit_latency_ms=self._avg(self._hit_latencies_ms),
                avg_miss_latency_ms=self._avg(self._miss_latencies_ms),
                total_cost_usd=self.total_cost_usd,
                total_baseline_cost_usd=self.total_baseline_cost_usd,
                total_savings_usd=self.total_baseline_cost_usd - self.total_cost_usd,
            )

    def as_dict(self) -> dict:
        """Snapshot as a plain dict (handy for JSON endpoints and Streamlit)."""
        return asdict(self.snapshot())
