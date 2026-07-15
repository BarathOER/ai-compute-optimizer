"""In-process metrics: request counts, latency, and estimated token cost.

Cost is estimated, not billed. Tokens are approximated from character length
(~4 chars/token, a standard rough heuristic). Input (prompt) and output
(completion) tokens are counted and priced *separately*, because real LLM
pricing charges them at different rates — output is several times more
expensive than input.

The key business metric is *savings*: for every request we compute what it
would have cost to always hit the remote model, and subtract what it actually
cost. Cache hits (cost 0) and local routes (cheaper backend) both contribute.
On top of the realized totals, :meth:`Metrics.snapshot` projects monthly and
annual savings from the measured hit rate and average per-query token usage.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field

MONTHS_PER_YEAR = 12


def estimate_tokens(text: str) -> int:
    """Approximate the token count of a string (~4 characters per token)."""
    return max(1, len(text) // 4)


@dataclass
class CostProjection:
    """Forward-looking savings estimate at an assumed monthly query volume.

    Projects the value of the *cache* specifically: at the measured hit rate,
    ``monthly_query_volume * hit_rate`` queries per month are served for free
    instead of costing the average remote price, versus a no-cache baseline.
    """

    monthly_query_volume: int
    hit_rate: float
    avg_input_tokens: float
    avg_output_tokens: float
    avg_remote_cost_per_query_usd: float
    projected_monthly_savings_usd: float
    projected_annual_savings_usd: float


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
    total_input_tokens: int
    total_output_tokens: int
    avg_input_tokens: float
    avg_output_tokens: float
    total_cost_usd: float
    total_baseline_cost_usd: float
    total_savings_usd: float
    projection: CostProjection


@dataclass
class Metrics:
    """Thread-safe accumulator for gateway metrics.

    Prices are per 1M tokens, split by input vs. output and by remote vs. local
    backend, matching how providers publish list prices.
    """

    remote_input_cost_per_1m: float
    remote_output_cost_per_1m: float
    local_input_cost_per_1m: float = 0.0
    local_output_cost_per_1m: float = 0.0
    monthly_query_volume: int = 0

    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    local_routes: int = 0
    remote_routes: int = 0
    _latencies_ms: list[float] = field(default_factory=list)
    _hit_latencies_ms: list[float] = field(default_factory=list)
    _miss_latencies_ms: list[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_baseline_cost_usd: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def cost_for(
        self, input_tokens: int, output_tokens: int, *, remote: bool
    ) -> float:
        """Return the estimated USD cost of a request on the given backend.

        Input and output tokens are priced at their own per-1M rates.
        """
        if remote:
            in_rate = self.remote_input_cost_per_1m
            out_rate = self.remote_output_cost_per_1m
        else:
            in_rate = self.local_input_cost_per_1m
            out_rate = self.local_output_cost_per_1m
        return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0

    def record(
        self,
        *,
        cache_hit: bool,
        route: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[float, float]:
        """Record one request and return ``(actual_cost, savings)`` for it.

        ``savings`` is the remote baseline cost minus the actual cost, i.e. what
        was avoided. On a cache hit the actual cost is 0, so the saving is the
        full price the remote call would have cost.
        """
        baseline_cost = self.cost_for(input_tokens, output_tokens, remote=True)
        if cache_hit:
            actual_cost = 0.0
        else:
            actual_cost = self.cost_for(
                input_tokens, output_tokens, remote=(route == "remote")
            )
        savings = baseline_cost - actual_cost

        with self._lock:
            self.total_requests += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
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

    def _project(
        self, hit_rate: float, avg_input: float, avg_output: float
    ) -> CostProjection:
        """Project cache-driven savings at the configured monthly volume.

        Baseline is "no cache": every query pays the average remote price. With
        the cache, a ``hit_rate`` fraction of queries cost nothing, so the
        monthly saving is ``volume * hit_rate * avg_remote_cost_per_query``.
        """
        avg_remote_cost = self.cost_for(
            round(avg_input), round(avg_output), remote=True
        )
        monthly = self.monthly_query_volume * hit_rate * avg_remote_cost
        return CostProjection(
            monthly_query_volume=self.monthly_query_volume,
            hit_rate=hit_rate,
            avg_input_tokens=avg_input,
            avg_output_tokens=avg_output,
            avg_remote_cost_per_query_usd=avg_remote_cost,
            projected_monthly_savings_usd=monthly,
            projected_annual_savings_usd=monthly * MONTHS_PER_YEAR,
        )

    def snapshot(self) -> MetricsSnapshot:
        """Return an immutable, serializable snapshot of current metrics."""
        with self._lock:
            total = self.total_requests
            hit_rate = self.cache_hits / total if total else 0.0
            avg_input = self.total_input_tokens / total if total else 0.0
            avg_output = self.total_output_tokens / total if total else 0.0
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
                total_input_tokens=self.total_input_tokens,
                total_output_tokens=self.total_output_tokens,
                avg_input_tokens=avg_input,
                avg_output_tokens=avg_output,
                total_cost_usd=self.total_cost_usd,
                total_baseline_cost_usd=self.total_baseline_cost_usd,
                total_savings_usd=self.total_baseline_cost_usd - self.total_cost_usd,
                projection=self._project(hit_rate, avg_input, avg_output),
            )

    def as_dict(self) -> dict:
        """Snapshot as a plain dict (handy for JSON endpoints and Streamlit)."""
        return asdict(self.snapshot())
