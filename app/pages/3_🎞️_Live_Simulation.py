"""Live Simulation page: animate a loaded policy on the 4x4 grid step by step."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.grid_renderer import (  # noqa: E402
    extract_state,
    render_grid,
    render_reward_curve,
)
from app.components.model_loader import list_checkpoints, load_model  # noqa: E402
from envs.traffic_grid_env import MultiIntersectionEnv  # noqa: E402

st.set_page_config(page_title="Live Simulation", page_icon="🎞️", layout="wide")
st.title("🎞️ Simulación en vivo")
st.caption("Cargá un modelo y mirá cómo controla la grilla 4×4 paso a paso.")


# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
st.sidebar.subheader("Modelo")

ckpts = list_checkpoints()
ckpt_options = ["(política aleatoria)"] + [
    str(p.relative_to(PROJECT_ROOT)) for p in ckpts
]
chosen = st.sidebar.selectbox("Checkpoint", ckpt_options)

uploaded = st.sidebar.file_uploader("…o subir un .pt", type=["pt"])

st.sidebar.subheader("Entorno")
demand = st.sidebar.slider("Demand factor", 0.0, 1.0, 1.0, 0.1)
episode_steps = st.sidebar.number_input("Pasos por episodio", 20, 2000, 200, step=20)
seed = st.sidebar.number_input("Seed", 0, 10_000, 42)

st.sidebar.subheader("Reproducción")
delay_ms = st.sidebar.slider("Delay entre pasos (ms)", 0, 500, 80, 20)
steps_per_click = st.sidebar.number_input("Pasos por clic en ▶", 1, 200, 50)


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------
def _new_episode():
    env = MultiIntersectionEnv(
        demand_factor=demand,
        episode_steps=int(episode_steps),
        seed=int(seed),
    )
    obs, _ = env.reset(seed=int(seed))
    st.session_state.env = env
    st.session_state.obs = obs
    st.session_state.step = 0
    st.session_state.cum_reward = 0.0
    st.session_state.rewards = []
    st.session_state.last_reward = 0.0
    st.session_state.done = False


if "env" not in st.session_state or st.sidebar.button("🔄 Reset episodio"):
    _new_episode()


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _cached_load(path: str):
    model, rms, _ = load_model(path)
    return model


model = None
model_label = "Random"
try:
    if uploaded is not None:
        tmp_path = PROJECT_ROOT / "checkpoints" / f"_uploaded_{uploaded.name}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(uploaded.getbuffer())
        model = _cached_load(str(tmp_path))
        model_label = f"Subido: {uploaded.name}"
    elif chosen != "(política aleatoria)":
        model = _cached_load(str(PROJECT_ROOT / chosen))
        model_label = chosen
except Exception as e:  # noqa: BLE001
    st.error(f"No pude cargar el modelo: {e}")


# ------------------------------------------------------------------
# Stepping
# ------------------------------------------------------------------
def _step_once():
    env = st.session_state.env
    obs = st.session_state.obs
    if model is None:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            actions, _, _ = model.act(obs_t)
            action = actions.squeeze(0).cpu().numpy()
    obs, r, term, trunc, _ = env.step(action)
    st.session_state.obs = obs
    st.session_state.step += 1
    st.session_state.cum_reward += float(r)
    st.session_state.last_reward = float(r)
    st.session_state.rewards.append(float(r))
    st.session_state.done = bool(term or trunc)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
top = st.container()
c1, c2, c3, c4 = top.columns(4)
c1.metric("Modelo", model_label.split("/")[-1] if "/" in model_label else model_label)
c2.metric("Paso", f"{st.session_state.step}/{int(episode_steps)}")
c3.metric("Reward acum.", f"{st.session_state.cum_reward:.2f}")
c4.metric("Último reward", f"{st.session_state.last_reward:.3f}")

ctrl1, ctrl2, ctrl3, _ = st.columns([1, 1, 1, 4])
do_step = ctrl1.button("⏭ 1 paso")
do_play = ctrl2.button(f"▶ {steps_per_click} pasos", type="primary")
do_full = ctrl3.button("⏩ Hasta el final")

grid_slot = st.empty()
curve_slot = st.empty()


def _refresh():
    queues, phases = extract_state(st.session_state.env)
    grid_slot.plotly_chart(
        render_grid(
            queues=queues,
            phases=phases,
            step=st.session_state.step,
            cumulative_reward=st.session_state.cum_reward,
            last_reward=st.session_state.last_reward,
            title_suffix=f" · {model_label}",
        ),
        use_container_width=True,
    )
    curve_slot.plotly_chart(
        render_reward_curve(st.session_state.rewards),
        use_container_width=True,
    )


# Run requested stepping
if do_step and not st.session_state.done:
    _step_once()
elif do_play and not st.session_state.done:
    for _ in range(int(steps_per_click)):
        if st.session_state.done:
            break
        _step_once()
        if delay_ms:
            time.sleep(delay_ms / 1000.0)
        _refresh()
elif do_full and not st.session_state.done:
    while not st.session_state.done:
        _step_once()

_refresh()

if st.session_state.done:
    st.success(
        f"Episodio finalizado en {st.session_state.step} pasos · "
        f"reward total = {st.session_state.cum_reward:.2f}"
    )
