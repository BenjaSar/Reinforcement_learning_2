"""Experiment manifest — JSON metadata for reproducibility."""

import os
import json
import time
import subprocess
from typing import Dict, Any, Optional


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception:
        return "unknown"


def save_manifest(
    config: Dict[str, Any],
    results: Dict[str, float],
    manifest_dir: str,
    seed: int,
    variant: str = "default",
) -> str:
    """Save experiment metadata as JSON. Returns path to manifest file."""
    os.makedirs(manifest_dir, exist_ok=True)

    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    filename = f"manifest_{variant}_seed{seed}_{timestamp.replace(':','-')}.json"
    filepath = os.path.join(manifest_dir, filename)

    manifest = {
        "experiment": {
            "variant": variant,
            "seed": seed,
            "timestamp": timestamp,
            "git_commit": get_git_commit(),
        },
        "config": {k: v for k, v in config.items() if not k.startswith("_")},
        "results": results,
    }

    with open(filepath, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return filepath
