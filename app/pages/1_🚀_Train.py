"""Train page: launch train_ppo.py / train_a2c.py with chosen hyperparameters."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Train", page_icon="🚀", layout="wide")
st.title("🚀 Entrenar un agente")
st.caption("Lanza `train_ppo.py` o `train_a2c.py` con los hiperparámetros que elijas.")

# ------------------------------------------------------------------
# Sidebar: algorithm + presets
# ------------------------------------------------------------------
algo = st.sidebar.radio("Algoritmo", ["PPO-GAE", "A2C (baseline)"])

preset = st.sidebar.selectbox(
    "Preset",
    ["⚡ Demo rápido (~30 s)", "🔧 Medio (~5 min)", "🏁 Completo (paper-like)"],
    index=0,
)

if "demo" in preset.lower():
    defaults = dict(n_updates=10, n_steps=32, n_envs=2, batch_size=64)
elif "medio" in preset.lower():
    defaults = dict(n_updates=100, n_steps=64, n_envs=4, batch_size=128)
else:
    defaults = dict(n_updates=500, n_steps=128, n_envs=16, batch_size=256)

# ------------------------------------------------------------------
# Form
# ------------------------------------------------------------------
with st.form("train_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        n_updates = st.number_input("n_updates", 1, 5000, defaults["n_updates"])
        n_steps = st.number_input("n_steps", 8, 1024, defaults["n_steps"])
        n_envs = st.number_input("n_envs", 1, 64, defaults["n_envs"])
        seed = st.number_input("seed", 0, 10_000, 0)
    with c2:
        gamma = st.slider("gamma (γ)", 0.80, 0.999, 0.99, 0.01)
        if algo.startswith("PPO"):
            gae_lambda = st.slider("gae_lambda (λ)", 0.80, 0.999, 0.95, 0.01)
            clip_epsilon = st.slider("clip_epsilon (ε)", 0.05, 0.5, 0.2, 0.05)
        else:
            gae_lambda = None
            clip_epsilon = None
        grad_clip = st.slider("grad_clip", 0.1, 2.0, 0.5, 0.1)
    with c3:
        if algo.startswith("PPO"):
            actor_lr = st.number_input("actor_lr", 1e-5, 1e-2, 3e-4, format="%.5f")
            critic_lr = st.number_input("critic_lr", 1e-5, 1e-2, 1e-3, format="%.5f")
            n_epochs = st.number_input("n_epochs (K)", 1, 20, 4)
            batch_size = st.number_input(
                "batch_size", 16, 4096, defaults["batch_size"]
            )
            curriculum = st.checkbox("Currículum 30→60→100%", value=False)
            c_e_start = st.number_input("c_e_start", 0.0, 0.5, 0.05, format="%.3f")
            c_e_end = st.number_input("c_e_end", 0.0, 0.5, 0.01, format="%.3f")
        else:
            lr = st.number_input("lr", 1e-5, 1e-2, 3e-4, format="%.5f")
            c_e = st.number_input("c_e (entropy)", 0.0, 0.5, 0.01, format="%.3f")

    submitted = st.form_submit_button("🚀 Lanzar entrenamiento", type="primary")

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if submitted:
    if algo.startswith("PPO"):
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train_ppo.py"),
            "--n_updates", str(n_updates),
            "--n_steps", str(n_steps),
            "--n_envs", str(n_envs),
            "--gamma", str(gamma),
            "--gae_lambda", str(gae_lambda),
            "--clip_epsilon", str(clip_epsilon),
            "--actor_lr", str(actor_lr),
            "--critic_lr", str(critic_lr),
            "--n_epochs", str(n_epochs),
            "--batch_size", str(batch_size),
            "--grad_clip", str(grad_clip),
            "--c_e_start", str(c_e_start),
            "--c_e_end", str(c_e_end),
            "--seed", str(seed),
        ]
        if curriculum:
            cmd.append("--curriculum")
        history_path = PROJECT_ROOT / "results" / "ppo" / "history.npy"
    else:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "train_a2c.py"),
            "--n_updates", str(n_updates),
            "--n_steps", str(n_steps),
            "--n_envs", str(n_envs),
            "--gamma", str(gamma),
            "--lr", str(lr),
            "--c_e", str(c_e),
            "--grad_clip", str(grad_clip),
            "--seed", str(seed),
        ]
        history_path = PROJECT_ROOT / "results" / "a2c" / "history.npy"

    st.code(" ".join(cmd), language="bash")

    log_box = st.empty()
    chart_slot = st.empty()
    progress = st.progress(0.0, text="Iniciando…")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding="utf-8",
        errors="replace",
    )

    log_lines: list[str] = []
    iter_count = 0
    t0 = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line.rstrip())
        if len(log_lines) > 250:
            log_lines = log_lines[-250:]
        if "iter " in line and "/" in line:
            try:
                # "[PPO] iter   12/100 |"  or  "[A2C] iter ..."
                segment = line.split("iter")[1].split("|")[0].strip()
                cur, tot = segment.split("/")
                iter_count = int(cur.strip())
                progress.progress(
                    min(iter_count / max(int(tot.strip()), 1), 1.0),
                    text=f"iter {iter_count}/{tot.strip()} · {time.time() - t0:.1f}s",
                )
            except Exception:
                pass
        log_box.code("\n".join(log_lines[-40:]), language="bash")

    proc.wait()
    progress.progress(1.0, text=f"Listo ({time.time() - t0:.1f}s)")

    if proc.returncode == 0:
        st.success("✅ Entrenamiento finalizado.")
    else:
        st.error(f"❌ Salió con código {proc.returncode}.")

    if history_path.exists():
        try:
            hist = np.load(history_path, allow_pickle=True).item()
            import plotly.graph_objects as go

            fig = go.Figure()
            if "avg_reward" in hist:
                fig.add_trace(
                    go.Scatter(
                        y=hist["avg_reward"], name="Avg Reward", line=dict(width=2)
                    )
                )
            for key in ("p_loss", "v_loss", "entropy"):
                if key in hist:
                    fig.add_trace(
                        go.Scatter(y=hist[key], name=key, line=dict(width=1, dash="dot"))
                    )
            fig.update_layout(
                title="Curvas de entrenamiento",
                height=420,
                paper_bgcolor="#0e1117",
                plot_bgcolor="#0e1117",
                font=dict(color="white"),
            )
            chart_slot.plotly_chart(fig, use_container_width=True)
        except Exception as e:  # noqa: BLE001
            st.warning(f"No pude graficar history.npy: {e}")

    img = history_path.parent / "training_curves.png"
    if img.exists():
        st.image(str(img), caption=str(img.relative_to(PROJECT_ROOT)))
