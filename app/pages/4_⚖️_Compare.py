"""Compare page: run N selected checkpoints against the env and plot a boxplot."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.model_loader import list_checkpoints, load_model  # noqa: E402
from envs.traffic_grid_env import MultiIntersectionEnv  # noqa: E402

st.set_page_config(page_title="Compare", page_icon="⚖️", layout="wide")
st.title("⚖️ Comparar checkpoints")
st.caption("Seleccioná varios `.pt`, corré N episodios y compará reward / cola.")

ckpts = list_checkpoints()
if not ckpts:
    st.warning("No hay checkpoints. Entrená al menos uno desde **🚀 Train**.")
    st.stop()

options = [str(p.relative_to(PROJECT_ROOT)) for p in ckpts]
selected = st.multiselect("Checkpoints a comparar", options, default=options[:2])

c1, c2, c3 = st.columns(3)
with c1:
    n_episodes = st.number_input("Episodios por checkpoint", 1, 50, 5)
with c2:
    episode_steps = st.number_input("Pasos por episodio", 50, 2000, 300, step=50)
with c3:
    demand = st.slider("Demand", 0.0, 1.0, 1.0, 0.1)

include_random = st.checkbox("Incluir baseline Random", value=True)
run = st.button("▶ Correr comparación", type="primary")


def _run_policy(model, env, n_episodes: int, seed_offset: int) -> list[dict]:
    out = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        total_r, qs = 0.0, []
        done = False
        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    actions, _, _ = model.act(obs_t)
                    action = actions.squeeze(0).cpu().numpy()
            obs, r, term, trunc, info = env.step(action)
            total_r += r
            qs.append(info["mean_queue"])
            done = term or trunc
        out.append({"reward": total_r, "mean_queue": float(np.mean(qs))})
    return out


if run and selected:
    progress = st.progress(0.0)
    results: dict[str, list[dict]] = {}

    def _new_env():
        return MultiIntersectionEnv(
            demand_factor=demand, episode_steps=int(episode_steps), seed=123
        )

    total_jobs = len(selected) + (1 if include_random else 0)
    done_jobs = 0

    if include_random:
        progress.progress(done_jobs / total_jobs, text="Random …")
        env = _new_env()
        results["Random"] = _run_policy(None, env, int(n_episodes), seed_offset=2000)
        env.close()
        done_jobs += 1

    for name in selected:
        progress.progress(done_jobs / total_jobs, text=f"{name} …")
        try:
            model, _, _ = load_model(str(PROJECT_ROOT / name))
            env = _new_env()
            results[name] = _run_policy(model, env, int(n_episodes), seed_offset=2000)
            env.close()
        except Exception as e:  # noqa: BLE001
            st.error(f"Fallo en {name}: {e}")
            results[name] = []
        done_jobs += 1
    progress.progress(1.0, text="Listo")

    rows = []
    for name, res in results.items():
        if not res:
            continue
        rs = [r["reward"] for r in res]
        qs = [r["mean_queue"] for r in res]
        rows.append(
            dict(
                policy=name,
                mean_reward=float(np.mean(rs)),
                std_reward=float(np.std(rs)),
                mean_queue=float(np.mean(qs)),
            )
        )
    st.subheader("Tabla")
    st.dataframe(rows, use_container_width=True)

    st.subheader("Distribución de reward (boxplot)")
    fig = go.Figure()
    for name, res in results.items():
        if not res:
            continue
        fig.add_trace(
            go.Box(
                y=[r["reward"] for r in res],
                name=name.split("/")[-1],
                boxpoints="all",
                jitter=0.4,
            )
        )
    fig.update_layout(
        height=460,
        yaxis_title="Reward por episodio",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cola promedio")
    bar = go.Figure(
        data=[
            go.Bar(
                x=[r["policy"].split("/")[-1] for r in rows],
                y=[r["mean_queue"] for r in rows],
                marker_color="#ff7f0e",
            )
        ]
    )
    bar.update_layout(
        height=360,
        yaxis_title="Mean queue (norm)",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    st.plotly_chart(bar, use_container_width=True)
