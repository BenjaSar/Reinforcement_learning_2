"""
MetricLogger — All Phases
===========================
Unified logging to TensorBoard + in-memory history.
Also provides matplotlib plot helpers for evaluation (Phase 4).
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
from typing import Dict, List, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    print("[Logger] TensorBoard not available. pip install tensorboard")


class MetricLogger:
    """
    Logs scalar metrics to TensorBoard and stores them in memory.

    Args:
        log_dir   : TensorBoard log directory
        algo_name : label prefix in TensorBoard
    """

    def __init__(self, log_dir: str = "results", algo_name: str = "PPO"):
        self.log_dir   = log_dir
        self.algo_name = algo_name
        self._history: Dict[str, List[float]] = {}

        os.makedirs(log_dir, exist_ok=True)
        if _TB_AVAILABLE:
            self._writer = SummaryWriter(log_dir=log_dir)
        else:
            self._writer = None

    # ------------------------------------------------------------------
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar metric."""
        full_tag = f"{self.algo_name}/{tag}"
        if self._writer is not None:
            self._writer.add_scalar(full_tag, value, step)
        if tag not in self._history:
            self._history[tag] = []
        self._history[tag].append(float(value))

    # ------------------------------------------------------------------
    def flush(self):
        if self._writer is not None:
            self._writer.flush()

    def close(self):
        if self._writer is not None:
            self._writer.close()

    # ------------------------------------------------------------------
    def get_history(self, tag: str) -> List[float]:
        return self._history.get(tag, [])

    def all_history(self) -> Dict[str, List[float]]:
        return dict(self._history)

    # ------------------------------------------------------------------
    # ---- Plot helpers (Phase 4) ----

    def _resolve_tag(self, tag: str) -> str:
        """
        Case-insensitive key lookup against self._history.
        Returns the exact stored key if found, otherwise returns tag unchanged.
        This makes plot_training_curves() robust against capitalisation
        differences between what callers request and what was logged
        (e.g. 'p_loss' vs 'P_Loss', 'avg_reward' vs 'Avg_Reward').
        """
        if tag in self._history:
            return tag
        tag_lower = tag.lower()
        for stored_key in self._history:
            if stored_key.lower() == tag_lower:
                return stored_key
        return tag   # not found — will produce an empty plot panel

    def plot_training_curves(
        self,
        save_path: str,
        metrics:   Optional[List[str]] = None,
        smooth:    int = 10,
    ):
        """
        Plot training curves for specified metrics.

        Args:
            save_path : output .png file path
            metrics   : list of metric names to plot (default: all available)
            smooth    : rolling average window size
        """
        # Default: plot everything that was actually logged
        if metrics is None:
            metrics = list(self._history.keys())

        # Resolve requested tags against actual stored keys (case-insensitive)
        resolved = [self._resolve_tag(m) for m in metrics]
        # Only keep tags that have data; skip unknown ones silently
        resolved = [t for t in resolved if len(self._history.get(t, [])) > 0]

        if len(resolved) == 0:
            print(f"[Logger] WARNING: No matching metrics found in history for {metrics}. "
                  f"Available keys: {list(self._history.keys())}")
            return

        n    = len(resolved)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        # Ensure axes is always a flat list regardless of grid shape
        if n == 1:
            axes = [axes]
        else:
            axes = np.array(axes).reshape(-1).tolist()

        for ax, tag in zip(axes, resolved):
            data = np.array(self._history[tag], dtype=float)
            x = np.arange(1, len(data) + 1)
            ax.plot(x, data, alpha=0.3, color="steelblue", label="raw")
            if len(data) >= smooth:
                smoothed = np.convolve(data, np.ones(smooth) / smooth, mode="valid")
                x_s = np.arange(smooth, len(data) + 1)
                ax.plot(x_s, smoothed, color="steelblue", linewidth=2,
                        label=f"MA({smooth})")
            ax.set_title(f"{self.algo_name} — {tag}")
            ax.set_xlabel("Update")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide any surplus axes in the last row
        for ax in axes[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Saved training curves -> {save_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def plot_comparison(
        histories:  Dict[str, Dict[str, List[float]]],
        metric:     str,
        save_path:  str,
        smooth:     int  = 10,
        title:      str  = "",
    ):
        """
        Overlay training curves for multiple algorithms on a single plot.

        Args:
            histories  : {algo_name: {metric_name: [values]}}
            metric     : which metric to compare
            save_path  : output .png file path
            smooth     : rolling average window
            title      : plot title
        """
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, (name, hist) in enumerate(histories.items()):
            data = np.array(hist.get(metric, []))
            if len(data) == 0:
                continue
            color = colors[i % len(colors)]
            x = np.arange(1, len(data) + 1)
            ax.plot(x, data, alpha=0.25, color=color)
            if len(data) >= smooth:
                smoothed = np.convolve(data, np.ones(smooth) / smooth, mode="valid")
                x_s = np.arange(smooth, len(data) + 1)
                ax.plot(x_s, smoothed, color=color, linewidth=2, label=name)

        ax.set_title(title or f"Comparison — {metric}")
        ax.set_xlabel("Update")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Saved comparison plot -> {save_path}")

    # ------------------------------------------------------------------
    @staticmethod
    def plot_multi_seed(
        seed_histories: List[Dict[str, List[float]]],
        metric:         str,
        save_path:      str,
        label:          str = "PPO",
        smooth:         int = 10,
    ):
        """
        Plot mean ± std band across multiple training seeds.

        Args:
            seed_histories : list of history dicts, one per seed
            metric         : which metric to plot
            save_path      : output .png file path
            label          : curve label
            smooth         : rolling average window
        """
        arrays = [
            np.array(h.get(metric, [])) for h in seed_histories
            if len(h.get(metric, [])) > 0
        ]
        if not arrays:
            return

        min_len = min(len(a) for a in arrays)
        stacked = np.stack([a[:min_len] for a in arrays])   # (S, T)

        mean = stacked.mean(axis=0)
        std  = stacked.std(axis=0)
        x    = np.arange(1, min_len + 1)

        if smooth > 1 and min_len >= smooth:
            kern   = np.ones(smooth) / smooth
            mean   = np.convolve(mean, kern, mode="valid")
            std    = np.convolve(std,  kern, mode="valid")
            x      = np.arange(smooth, min_len + 1)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, mean, linewidth=2, label=f"{label} (mean)")
        ax.fill_between(x, mean - std, mean + std, alpha=0.2,
                        label=f"{label} (±1 std)")
        ax.set_title(f"{label} — {metric} across {len(arrays)} seeds")
        ax.set_xlabel("Update")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Logger] Saved multi-seed plot -> {save_path}")
