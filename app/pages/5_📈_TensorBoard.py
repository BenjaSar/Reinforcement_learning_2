"""TensorBoard page: launch TB as a subprocess and embed via iframe."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="TensorBoard", page_icon="📈", layout="wide")
st.title("📈 TensorBoard embebido")
st.caption("Lanza TensorBoard en background apuntando a `results/` y lo muestra acá.")

PORT = st.sidebar.number_input("Puerto", 6006, 6100, 6006)


def _tb_running(port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        try:
            s.connect(("127.0.0.1", int(port)))
            return True
        except OSError:
            return False


col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Iniciar TensorBoard", type="primary"):
        if _tb_running(PORT):
            st.info(f"Ya hay un servicio en el puerto {PORT}.")
        else:
            try:
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "tensorboard.main",
                        "--logdir",
                        str(RESULTS_DIR),
                        "--port",
                        str(PORT),
                        "--host",
                        "127.0.0.1",
                    ],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                with st.spinner("Levantando TensorBoard…"):
                    for _ in range(20):
                        if _tb_running(PORT):
                            break
                        time.sleep(0.5)
                if _tb_running(PORT):
                    st.success(f"TensorBoard arriba en :{PORT}")
                else:
                    st.error(
                        "No pude verificar TensorBoard. Probá instalarlo: "
                        "`pip install tensorboard`."
                    )
            except FileNotFoundError:
                st.error("`tensorboard` no está instalado: `pip install tensorboard`.")
with col2:
    st.write(f"Estado: {'🟢 corriendo' if _tb_running(PORT) else '⚪ detenido'}")
    st.write(f"URL: http://127.0.0.1:{PORT}")

st.divider()
if _tb_running(PORT):
    components.iframe(f"http://127.0.0.1:{PORT}", height=820, scrolling=True)
else:
    st.info(
        "Aún no detecto TensorBoard. Hacé clic en **🚀 Iniciar TensorBoard**.\n\n"
        "Si no lo tenés instalado: `pip install tensorboard`."
    )
