"""Helpers to discover, load and run PPO/A2C checkpoints from the Streamlit UI."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.actor_critic import SharedActorCritic
from utils.running_stats import RunningMeanStd


CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def list_checkpoints() -> List[Path]:
    """Return all .pt files found under checkpoints/."""
    if not CHECKPOINTS_DIR.exists():
        return []
    return sorted(p for p in CHECKPOINTS_DIR.rglob("*.pt"))


def load_model(ckpt_path: str | Path, device: str = "cpu"):
    """Load a SharedActorCritic from a checkpoint .pt file.

    Returns (model, rms_or_None, raw_ckpt_dict).
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = SharedActorCritic()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    rms: Optional[RunningMeanStd] = None
    if isinstance(ckpt, dict) and "rms_mean" in ckpt:
        rms = RunningMeanStd(shape=())
        rms.mean = np.float64(ckpt["rms_mean"])
        rms.var = np.float64(ckpt["rms_var"])
        rms.count = np.float64(ckpt["rms_count"])

    model.eval()
    return model, rms, ckpt


def load_scripted_policy(ckpt_path: str | Path):
    """Load a TorchScript-exported policy (.pt)."""
    return torch.jit.load(str(ckpt_path), map_location="cpu")
