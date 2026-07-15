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
    # --- Billing cluster: same intent, different wording ---
    "A customer is asking why their subscription renewed at a higher price than "
    "last year. They signed up during a promotional period that has now ended. "
    "Draft a clear, empathetic response explaining the pricing change, what the "
    "promotional rate covered, and what options they have going forward.",

    "Why did my subscription cost go up after renewal? I signed up on a promo "
    "deal and now I'm being charged more. Please explain what happened and tell "
    "me what my options are if I want a cheaper plan.",

    "Customer complaint: their bill increased at renewal because their intro "
    "discount expired. Write a supportive reply that explains the promotional "
    "pricing ended, breaks down the new charge, and outlines alternatives.",

    # --- Password/access cluster ---
    "A user cannot log into their account after enabling two-factor "
    "authentication. They no longer have access to the phone number they "
    "registered. Write a step-by-step guide explaining how they can recover "
    "access, what verification we require, and how long the process takes.",

    "I turned on 2FA and now I'm locked out because I changed my phone number. "
    "How do I get back into my account? What proof do you need from me and how "
    "long will it take to sort out?",

    # --- Integration cluster ---
    "Explain to a non-technical customer how to connect our platform to their "
    "existing CRM. Cover what permissions are needed, roughly how long setup "
    "takes, what data syncs, and what to do if the connection fails.",

    "How do I link my CRM to your product? I'm not technical. What access do "
    "you need, how long does it take, which data comes across, and what should "
    "I do if it doesn't work?",
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
