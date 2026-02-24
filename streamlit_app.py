import pandas as pd
import streamlit as st
import os

st.set_page_config(page_title="Monitoreo Modelo", layout="wide")

st.title("📊 Dashboard Monitoreo Modelo ML")
st.caption("Sistema de detección de Data Drift en producción")

root = os.path.dirname(__file__)

latest_path = os.path.join(root, "latest_drift_metrics.csv")
log_path = os.path.join(root, "monitoring_log.csv")

# =============================
# Función colores semáforo
# =============================
def color_severity(val):
    if val == "ROJO":
        return "background-color:#ff4d4d;color:white"
    elif val == "AMARILLO":
        return "background-color:#ffd11a;color:black"
    else:
        return "background-color:#85e085;color:black"

# =============================
# Tabla principal
# =============================
st.subheader("Estado actual de variables")

if os.path.exists(latest_path):

    df = pd.read_csv(latest_path)

    st.dataframe(
        df.style.applymap(color_severity, subset=["severity"]),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Variables críticas", (df["severity"]=="ROJO").sum())
    col2.metric("Variables alerta", (df["severity"]=="AMARILLO").sum())
    col3.metric("Variables estables", (df["severity"]=="VERDE").sum())

else:
    st.error("No existe archivo de métricas. Ejecuta primero model_monitoring.py")

# =============================
# Evolución temporal
# =============================
st.subheader("Evolución temporal del drift")

if os.path.exists(log_path):

    log_df = pd.read_csv(log_path)

    st.line_chart(
        log_df.set_index("period")[["avg_psi","pct_red"]]
    )

else:
    st.info("No existe historial temporal")

# =============================
# Recomendación automática
# =============================
st.subheader("Diagnóstico automático")

if os.path.exists(latest_path):

    if (df["severity"]=="ROJO").any():
        st.error("⚠ Drift crítico detectado → Se recomienda reentrenar modelo")
    elif (df["severity"]=="AMARILLO").any():
        st.warning("Cambios detectados → Monitorear evolución")
    else:
        st.success("Modelo estable ✔")