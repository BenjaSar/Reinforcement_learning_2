"""Plotly renderer for the 4x4 traffic intersection grid."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

GRID_SIZE = 4
N_INTERSECTIONS = 16
N_LANES = 4
N_PHASES = 4
LANE_CAPACITY = 20.0

PHASE_LABELS = ["N-S", "E-W", "NE-SW", "NW-SE"]
PHASE_ARROWS = ["↕", "↔", "⤢", "⤡"]


def queues_color(mean_q: float) -> str:
    if mean_q < 0.15:
        return "#2ca02c"
    if mean_q < 0.35:
        return "#ffb000"
    if mean_q < 0.6:
        return "#ff7f0e"
    return "#d62728"


def render_grid(
    queues: np.ndarray,
    phases: np.ndarray,
    step: int,
    cumulative_reward: float,
    last_reward: float,
    title_suffix: str = "",
) -> go.Figure:
    """Render a 4x4 grid heatmap with annotations.

    Args:
        queues: shape (16, 4) — queue lengths per lane (0..LANE_CAPACITY)
        phases: shape (16,)  — active phase per intersection (0..3)
        step:   current simulation step
        cumulative_reward: cumulative reward so far in the episode
        last_reward: reward at the current step
    """
    norm = queues / LANE_CAPACITY
    mean_per_inter = norm.mean(axis=1).reshape(GRID_SIZE, GRID_SIZE)

    heat = go.Heatmap(
        z=mean_per_inter,
        zmin=0.0,
        zmax=1.0,
        colorscale=[
            [0.00, "#1a9641"],
            [0.25, "#a6d96a"],
            [0.50, "#fdae61"],
            [0.75, "#f46d43"],
            [1.00, "#d7191c"],
        ],
        colorbar=dict(title="Carga", thickness=14, len=0.75),
        hovertemplate="Inter (%{x},%{y})<br>Carga media: %{z:.2f}<extra></extra>",
        showscale=True,
    )

    annotations = []
    for i in range(N_INTERSECTIONS):
        row = i // GRID_SIZE
        col = i % GRID_SIZE
        ph = int(phases[i])
        q = queues[i]
        text = (
            f"<b>I{i:02d}</b> {PHASE_ARROWS[ph]}<br>"
            f"<span style='font-size:9px'>"
            f"N:{q[0]:.0f} E:{q[1]:.0f}<br>"
            f"S:{q[2]:.0f} O:{q[3]:.0f}"
            f"</span>"
        )
        annotations.append(
            dict(
                x=col,
                y=row,
                text=text,
                showarrow=False,
                font=dict(color="white", size=11, family="monospace"),
                align="center",
            )
        )

    fig = go.Figure(data=[heat])
    fig.update_layout(
        annotations=annotations,
        title=dict(
            text=(
                f"<b>Grilla 4×4 · paso {step}</b>"
                f"<br><span style='font-size:12px'>"
                f"Reward acumulado: {cumulative_reward:.2f} "
                f"· último: {last_reward:.3f}{title_suffix}"
                f"</span>"
            ),
            x=0.5,
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(GRID_SIZE)),
            ticktext=[f"col {i}" for i in range(GRID_SIZE)],
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(GRID_SIZE)),
            ticktext=[f"fila {i}" for i in range(GRID_SIZE)],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=520,
        margin=dict(l=40, r=40, t=80, b=40),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def render_reward_curve(rewards: list[float]) -> go.Figure:
    cum = np.cumsum(rewards) if rewards else np.array([])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=rewards,
            mode="lines",
            name="Reward por paso",
            line=dict(color="#1f77b4", width=1.2),
        )
    )
    fig.add_trace(
        go.Scatter(
            y=cum / max(len(rewards), 1),
            mode="lines",
            name="Reward medio (acum)",
            line=dict(color="#2ca02c", width=2),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=40, r=40, t=20, b=30),
        xaxis=dict(title="Paso"),
        yaxis=dict(title="Reward step", side="left"),
        yaxis2=dict(title="Reward medio", side="right", overlaying="y"),
        legend=dict(orientation="h", y=-0.25),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    return fig


def extract_state(env) -> tuple[np.ndarray, np.ndarray]:
    """Pull (queues, phases) from a MultiIntersectionEnv instance."""
    return env._queues.copy(), env._current_phase.copy()
