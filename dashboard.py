"""Streamlit dashboard for the AI Compute Optimizer.

Polls the gateway's ``/metrics`` and ``/health`` endpoints and visualizes the
three numbers that justify the gateway's existence: cache hit rate, latency
(hit vs miss), and cumulative cost saved.

Run with::

    streamlit run dashboard.py

Set ``GATEWAY_URL`` to point at a non-default gateway (default
``http://localhost:8000``).
"""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")


def fetch_metrics(url: str) -> dict | None:
    """Fetch the metrics snapshot from the gateway, or None if unreachable."""
    try:
        response = httpx.get(f"{url}/metrics", timeout=5.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        return None


def main() -> None:
    st.set_page_config(page_title="AI Compute Optimizer", page_icon="⚡", layout="wide")
    st.title("⚡ AI Compute Optimizer — Live Dashboard")
    st.caption(f"Gateway: {GATEWAY_URL}")

    with st.sidebar:
        st.header("Settings")
        url = st.text_input("Gateway URL", value=GATEWAY_URL)
        if st.button("Refresh"):
            st.rerun()

    data = fetch_metrics(url)
    if data is None:
        st.error(f"Could not reach the gateway at {url}. Is it running?")
        st.stop()

    # --- Headline metrics -----------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total requests", f"{data['total_requests']}")
    col2.metric("Cache hit rate", f"{data['hit_rate'] * 100:.1f}%")
    col3.metric("Total cost", f"${data['total_cost_usd']:.4f}")
    col4.metric(
        "Total saved",
        f"${data['total_savings_usd']:.4f}",
        help="Estimated cost avoided vs. always calling the remote model.",
    )

    st.divider()

    # --- Latency: hit vs miss -------------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Latency (avg ms)")
        latency_df = pd.DataFrame(
            {
                "phase": ["cache hit", "cache miss", "overall"],
                "latency_ms": [
                    data["avg_hit_latency_ms"],
                    data["avg_miss_latency_ms"],
                    data["avg_latency_ms"],
                ],
            }
        ).set_index("phase")
        st.bar_chart(latency_df)

    with right:
        st.subheader("Request routing")
        routing_df = pd.DataFrame(
            {
                "route": ["cache hit", "local", "remote"],
                "count": [
                    data["cache_hits"],
                    data["local_routes"],
                    data["remote_routes"],
                ],
            }
        ).set_index("route")
        st.bar_chart(routing_df)

    st.divider()

    # --- Cost breakdown --------------------------------------------------
    st.subheader("Cost: actual vs all-remote baseline")
    cost_df = pd.DataFrame(
        {
            "scenario": ["actual cost", "baseline (all remote)"],
            "usd": [data["total_cost_usd"], data["total_baseline_cost_usd"]],
        }
    ).set_index("scenario")
    st.bar_chart(cost_df)

    with st.expander("Raw metrics"):
        st.json(data)


if __name__ == "__main__":
    main()
