"""Benchmark cache-hit vs cache-miss latency and estimated cost savings.

Sends a set of prompts to a running gateway twice: the first pass populates the
cache (misses), the second pass should be served from cache (hits). It then
reports latency percentiles for each phase and the total estimated savings.

Usage::

    # start the gateway first, e.g. `uvicorn app.main:app --reload`
    python benchmark.py --url http://localhost:8000 --rounds 2

The script talks to the HTTP API only, so it works against a local process or a
containerized deployment without importing the app.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import httpx

DEFAULT_PROMPTS: list[str] = [
    "What is the capital of France?",
    "Explain list comprehensions in Python.",
    "What is the boiling point of water at sea level?",
    "Summarize the theory of relativity in one sentence.",
    "How do I reverse a string in Python?",
    "What is the largest planet in the solar system?",
]


@dataclass
class PhaseResult:
    """Aggregated results for one benchmark phase."""

    label: str
    latencies_ms: list[float]
    cache_hits: int
    total_cost_usd: float
    total_savings_usd: float

    @property
    def count(self) -> int:
        return len(self.latencies_ms)

    @property
    def avg_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]


def _run_phase(client: httpx.Client, url: str, label: str, prompts: list[str]) -> PhaseResult:
    """Send every prompt once and collect latency/cost from the responses."""
    latencies: list[float] = []
    hits = 0
    cost = 0.0
    savings = 0.0
    for prompt in prompts:
        started = time.perf_counter()
        response = client.post(f"{url}/query", json={"prompt": prompt})
        response.raise_for_status()
        wall_ms = (time.perf_counter() - started) * 1000.0
        body = response.json()
        latencies.append(wall_ms)
        hits += int(body["cache_hit"])
        cost += float(body["estimated_cost_usd"])
        savings += float(body["estimated_savings_usd"])
    return PhaseResult(label, latencies, hits, cost, savings)


def _print_phase(result: PhaseResult) -> None:
    print(f"\n[{result.label}]")
    print(f"  requests     : {result.count}")
    print(f"  cache hits   : {result.cache_hits}/{result.count}")
    print(f"  avg latency  : {result.avg_ms:8.2f} ms")
    print(f"  p95 latency  : {result.p95_ms:8.2f} ms")
    print(f"  actual cost  : ${result.total_cost_usd:.6f}")
    print(f"  est. savings : ${result.total_savings_usd:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000", help="Gateway base URL.")
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of passes over the prompt set (>=2 to see hits).",
    )
    args = parser.parse_args()

    prompts = DEFAULT_PROMPTS
    print(f"Benchmarking {args.url} with {len(prompts)} prompts x {args.rounds} rounds")

    phases: list[PhaseResult] = []
    with httpx.Client(timeout=120.0) as client:
        for round_index in range(args.rounds):
            label = "miss (cold)" if round_index == 0 else f"hit (round {round_index + 1})"
            phases.append(_run_phase(client, args.url, label, prompts))

    for phase in phases:
        _print_phase(phase)

    if len(phases) >= 2:
        cold, warm = phases[0], phases[1]
        if warm.avg_ms > 0:
            speedup = cold.avg_ms / warm.avg_ms if warm.avg_ms else float("inf")
            print("\n=== Summary ===")
            print(f"  latency speedup (cold/warm): {speedup:6.2f}x")
        total_savings = sum(p.total_savings_usd for p in phases)
        total_baseline = total_savings + sum(p.total_cost_usd for p in phases)
        pct = (total_savings / total_baseline * 100.0) if total_baseline else 0.0
        print(f"  total estimated savings    : ${total_savings:.6f} ({pct:.1f}% vs all-remote)")


if __name__ == "__main__":
    main()
