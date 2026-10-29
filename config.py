from dataclasses import dataclass, field
# USD per 1M tokens (input, output). ESTIMATES - verify against current OpenAI pricing and edit.
PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4.1-mini": (0.40, 1.60),
          "gpt-4.1": (2.0, 8.0), "gpt-4o": (2.5, 10.0)}
MODELS = list(PRICES)
STAGES = ["analyser", "planner", "researcher", "synthesis", "qc"]
MODELS0 = {"analyser": "gpt-4o-mini", "planner": "gpt-4.1", "researcher": "gpt-4o-mini",
           "synthesis": "gpt-4.1", "qc": "gpt-4.1-mini"}
TEMPS0 = {"analyser": 0.2, "planner": 0.2, "researcher": 0.1, "synthesis": 0.4, "qc": 0.0}

@dataclass
class Settings:
    models: dict = field(default_factory=lambda: dict(MODELS0))
    temps: dict = field(default_factory=lambda: dict(TEMPS0))
    max_iterations: int = 40      # model calls per pass
    max_tool_calls: int = 60      # dispatches per pass
    budget_usd: float = 1.0       # cumulative per run
    max_steps_per_q: int = 5
    max_repair_rounds: int = 2
    timeout: float = 60.0
    tool_timeout: float = 10.0
    max_retries: int = 4
