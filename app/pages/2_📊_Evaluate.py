"""Evaluate page: run evaluate.py or in-process evaluation with chosen checkpoints."""

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

from app.components.model_loader import list_checkpoints  # noqa: E402

st.set_page_config(page_title="Evaluate", page_icon="📊", layout="wide")
st.title("📊 Evaluar checkpoints")
st.caption("Corre `evaluate.py` para comparar A2C / PPO contra Random, Fixed-Time y Actuated.")

ckpts = list_checkpoints()
if not ckpts:
    st.warning(
        "No hay checkpoints. Andá a **🚀 Train** primero o copiá un `.pt` "
        "manualmente a `checkpoints/`."
    )
    st.stop()

ckpt_strs = [str(p.relative_to(PROJECT_ROOT)) for p in ckpts]

c1, c2 = st.columns(2)
with c1:
    a2c_pick = st.selectbox(
        "Checkpoint A2C",
        ["(ninguno)"] + [s for s in ckpt_strs if "a2c" in s.lower()],
    )
with c2:
    ppo_pick = st.selectbox(
        "Checkpoint PPO",
        ["(ninguno)"] + [s for s in ckpt_strs if "ppo" in s.lower()],
    )

c3, c4, c5 = st.columns(3)
with c3:
    n_seeds = st.number_input("n_seeds", 1, 20, 3)
with c4:
    n_eval_episodes = st.number_input("Episodios por seed", 1, 100, 5)
with c5:
    demand = st.slider("Demand factor", 0.0, 1.0, 1.0, 0.1)

run = st.button("📊 Ejecutar evaluación", type="primary")

if run:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--n_seeds", str(n_seeds),
        "--n_eval_episodes", str(n_eval_episodes),
        "--demand", str(demand),
    ]
    if a2c_pick != "(ninguno)":
        cmd += ["--a2c_ckpt", a2c_pick]
    else:
        cmd += ["--a2c_ckpt", "checkpoints/_skip_a2c.pt"]
    if ppo_pick != "(ninguno)":
        cmd += ["--ppo_ckpt", ppo_pick]
    else:
        cmd += ["--ppo_ckpt", "checkpoints/_skip_ppo.pt"]

    st.code(" ".join(cmd), language="bash")

    log_box = st.empty()
    log_lines: list[str] = []
    t0 = time.time()

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
    assert proc.stdout is not None
    for line in proc.stdout:
        log_lines.append(line.rstrip())
        log_box.code("\n".join(log_lines[-60:]), language="bash")
    proc.wait()
    st.write(f"⏱️ {time.time() - t0:.1f}s — exit {proc.returncode}")

    eval_dir = PROJECT_ROOT / "results" / "eval"
    table = eval_dir / "comparison_table.txt"
    if table.exists():
        st.subheader("Tabla comparativa")
        st.code(table.read_text(encoding="utf-8"))

    for img_name, caption in [
        ("reward_comparison.png", "Comparación de reward medio"),
        ("queue_comparison.png", "Comparación de cola promedio"),
        ("ppo_seed_distribution.png", "Distribución PPO por seed"),
    ]:
        img = eval_dir / img_name
        if img.exists():
            st.image(str(img), caption=caption)

st.divider()
st.subheader("Última tabla guardada")
existing_table = PROJECT_ROOT / "results" / "eval" / "comparison_table.txt"
if existing_table.exists():
    st.code(existing_table.read_text(encoding="utf-8"))
else:
    st.info("Todavía no hay evaluación guardada.")
