"""AI Compute Optimizer — an LLM cost-reduction API gateway.

A FastAPI service that fronts one or more LLMs with a semantic cache and a
complexity-based router, reducing token spend by (a) serving semantically
similar prompts from cache and (b) sending simple prompts to a cheap local
model while reserving a paid frontier model for genuinely complex ones.
"""

__version__ = "1.0.0"
