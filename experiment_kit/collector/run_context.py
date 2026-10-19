"""Reader and basic validation for optional offline run-context evidence."""
from __future__ import annotations

import json
from pathlib import Path


def load_run_context(path: Path) -> dict:
    """Load a run-context JSON object without contacting external systems."""
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Run context path is not a file")
    with path.open("r", encoding="utf-8") as stream:
        context = json.load(stream)
    if not isinstance(context, dict):
        raise ValueError("Run context must be a JSON object")
    for key in ("boundaries", "workload", "environment"):
        if key in context and not isinstance(context[key], dict):
            raise ValueError(f"Run context {key} must be an object")
    if "historical_audits" in context and not isinstance(context["historical_audits"], list):
        raise ValueError("Run context historical_audits must be a list")
    if "identity_samples" in context and not isinstance(context["identity_samples"], list):
        raise ValueError("Run context identity_samples must be a list")
    return context
