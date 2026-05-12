"""
RunningMeanStd — Phase 3 / 5
==============================
Online running mean and standard deviation for reward normalization.

Based on Welford's online algorithm.
Reference: Stable Baselines3 implementation.
"""

import numpy as np


class RunningMeanStd:
    """
    Maintains a running estimate of mean and variance for a stream of values.

    Supports batched updates (batch axis=0).

    Args:
        shape  : shape of a single sample (use () for scalars)
        epsilon: small constant for numerical stability
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-4):
        self.mean    = np.zeros(shape, dtype=np.float64)
        self.var     = np.ones(shape,  dtype=np.float64)
        self.count   = epsilon

    # ------------------------------------------------------------------
    def update(self, x: np.ndarray):
        """Update statistics with a batch of observations."""
        x = np.asarray(x, dtype=np.float64)
        batch_mean  = x.mean(axis=0)
        batch_var   = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean:  np.ndarray,
        batch_var:   np.ndarray,
        batch_count: int,
    ):
        delta       = batch_mean - self.mean
        tot_count   = self.count + batch_count

        new_mean    = self.mean + delta * batch_count / tot_count
        m_a         = self.var * self.count
        m_b         = batch_var * batch_count
        m2          = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var     = m2 / tot_count

        self.mean   = new_mean
        self.var    = new_var
        self.count  = tot_count

    # ------------------------------------------------------------------
    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize array using running statistics."""
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return (f"RunningMeanStd(mean={self.mean:.4f}, "
                f"std={float(self.std):.4f}, n={self.count:.0f})")
