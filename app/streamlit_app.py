"""Streamlit entry point for the PPO-GAE Traffic Signal Control UI.

Run from the project root:

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.model_loader import list_checkpoints  # noqa: E402

st.set_page_config(
    page_title="PPO-GAE Traffic Control",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚦 PPO-GAE Traffic Signal Control")
st.caption("Reinforcement Learning II · TP Final · Interfaz interactiva")

st.markdown(
    """
Esta UI envuelve el proyecto **PPO-GAE / A2C** sobre la grilla 4×4 de
intersecciones. Desde aquí podés:

- **🚀 Entrenar** un modelo A2C o PPO con sliders de hiperparámetros.
- **📊 Evaluar** checkpoints contra baselines (Random / Fixed-Time / Actuated).
- **🎞️ Simular en vivo** un modelo cargado y ver la grilla paso a paso.
- **⚖️ Comparar** múltiples checkpoints entre sí.
- **📈 TensorBoard** embebido para curvas detalladas.

Usá el menú de la izquierda para navegar entre páginas.
"""
)

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Resumen del proyecto")
    st.markdown(
        """
| Capa | Qué hace |
|------|---------|
| **Entorno** (`envs/`) | 4×4 = 16 intersecciones · estado 144-dim · acción `MultiDiscrete([4]*16)` · llegadas Poisson · min-green 5 pasos |
| **Modelo** (`models/`) | SharedActorCritic: tronco compartido + 16 cabezas actor + crítica escalar |
| **Algoritmos** (`algorithms/`) | A2C baseline · PPO-GAE con clip ε=0.2, K=4 epochs, KL early-stop |
| **Utilidades** (`utils/`) | Normalización de rewards · currículum 30→60→100% · logger TensorBoard |
| **Scripts** | `train_a2c.py` · `train_ppo.py` · `evaluate.py` · `sweep.py` |
"""
    )

with col2:
    st.subheader("Checkpoints encontrados")
    ckpts = list_checkpoints()
    if not ckpts:
        st.warning(
            "No hay checkpoints todavía.\n\n"
            "Andá a **🚀 Train** para entrenar un modelo o copiá un `.pt` "
            "a `checkpoints/`."
        )
    else:
        for p in ckpts:
            rel = p.relative_to(PROJECT_ROOT)
            size_kb = p.stat().st_size / 1024
            st.markdown(f"- `{rel}` · {size_kb:.1f} KB")

st.divider()

st.subheader("Pipeline de las 5 fases")
st.markdown(
    """
1. **Phase 1** — Entorno custom de tráfico (`MultiIntersectionEnv`).
2. **Phase 2** — Baseline A2C (`train_a2c.py`).
3. **Phase 3** — PPO-GAE con 8 mejoras (`train_ppo.py`).
4. **Phase 4** — Evaluación multi-seed + sweep (`evaluate.py`, `sweep.py`).
5. **Phase 5** — Currículum, TensorBoard, checkpointing, TorchScript export.
"""
)

st.info(
    "💡 Tip: si nunca corriste el proyecto, empezá por **🚀 Train** con un "
    "preset rápido (5–20 updates) para generar un checkpoint, después usá "
    "**🎞️ Live Simulation** para ver al agente actuar."
)
