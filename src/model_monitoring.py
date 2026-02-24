import os
import json
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# =========================
# Utilidades de Drift
# =========================

EPS = 1e-8

def _safe_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, None)
    return p / p.sum()

def ks_statistic(x_ref: np.ndarray, x_cur: np.ndarray) -> float:
    """KS D-statistic (sin p-value) implementado sin scipy."""
    x_ref = np.asarray(x_ref, dtype=float)
    x_cur = np.asarray(x_cur, dtype=float)
    x_ref = x_ref[~np.isnan(x_ref)]
    x_cur = x_cur[~np.isnan(x_cur)]
    if len(x_ref) == 0 or len(x_cur) == 0:
        return np.nan
    data_all = np.sort(np.unique(np.concatenate([x_ref, x_cur])))
    cdf_ref = np.searchsorted(np.sort(x_ref), data_all, side="right") / len(x_ref)
    cdf_cur = np.searchsorted(np.sort(x_cur), data_all, side="right") / len(x_cur)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))

def psi_from_bins(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """PSI usando conteos por bins ya definidos."""
    ref = _safe_prob(ref_counts)
    cur = _safe_prob(cur_counts)
    return float(np.sum((cur - ref) * np.log((cur + EPS) / (ref + EPS))))

def jensen_shannon_divergence(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """JS divergence usando distribuciones discretas (bins o categorías)."""
    p = _safe_prob(ref_counts)
    q = _safe_prob(cur_counts)
    m = 0.5 * (p + q)

    def kl(a, b):
        a = np.clip(a, EPS, None)
        b = np.clip(b, EPS, None)
        return np.sum(a * np.log(a / b))

    js = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(js)

def cramers_v_from_crosstab(ct: pd.DataFrame) -> float:
    """Cramér's V (0-1) a partir de tabla de contingencia."""
    obs = ct.values.astype(float)
    n = obs.sum()
    if n == 0:
        return np.nan
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    chi2 = np.nansum((obs - expected) ** 2 / (expected + EPS))

    r, k = obs.shape
    if min(r - 1, k - 1) == 0:
        return np.nan
    return float(math.sqrt((chi2 / n) / (min(r - 1, k - 1))))

def make_numeric_bins(x_ref: pd.Series, n_bins: int = 10) -> np.ndarray:
    """Bins por cuantiles para estabilidad (menos sensible a outliers)."""
    x = pd.to_numeric(x_ref, errors="coerce")
    x = x.dropna()
    if x.empty:
        return np.array([0, 1], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(x, qs))
    if len(edges) < 2:
        edges = np.array([x.min(), x.max() + 1e-6], dtype=float)
    return edges

def bin_counts(x: pd.Series, edges: np.ndarray) -> np.ndarray:
    x = pd.to_numeric(x, errors="coerce")
    counts, _ = np.histogram(x.dropna().values, bins=edges)
    return counts

def drift_severity(psi: float, ks: float, js: float, cv: float) -> str:
    """
    Semáforo simple (ajustable):
    - Rojo si PSI>=0.3 o KS>=0.2 o JS>=0.15 o CramersV>=0.3
    - Amarillo si PSI>=0.2 o KS>=0.1 o JS>=0.1 o CramersV>=0.2
    - Verde en caso contrario
    """
    if (psi is not None and psi >= 0.30) or (ks is not None and ks >= 0.20) or (js is not None and js >= 0.15) or (cv is not None and cv >= 0.30):
        return "ROJO"
    if (psi is not None and psi >= 0.20) or (ks is not None and ks >= 0.10) or (js is not None and js >= 0.10) or (cv is not None and cv >= 0.20):
        return "AMARILLO"
    return "VERDE"

# =========================
# Perfil baseline
# =========================

def build_baseline_profile(df_ref: pd.DataFrame, numeric_cols, cat_cols, n_bins=10) -> dict:
    profile = {"created_at": datetime.utcnow().isoformat(), "numeric": {}, "categorical": {}}

    for col in numeric_cols:
        edges = make_numeric_bins(df_ref[col], n_bins=n_bins)
        ref_counts = bin_counts(df_ref[col], edges).tolist()
        profile["numeric"][col] = {"edges": edges.tolist(), "ref_counts": ref_counts}

    for col in cat_cols:
        vc = df_ref[col].astype(str).fillna("MISSING").value_counts()
        profile["categorical"][col] = {"ref_counts": vc.to_dict()}

    return profile

def compute_drift(df_ref: pd.DataFrame, df_cur: pd.DataFrame, profile: dict, numeric_cols, cat_cols) -> pd.DataFrame:
    rows = []

    # Numeric drift
    for col in numeric_cols:
        info = profile["numeric"].get(col)
        if not info:
            continue
        edges = np.array(info["edges"], dtype=float)
        ref_counts = np.array(info["ref_counts"], dtype=float)
        cur_counts = bin_counts(df_cur[col], edges)

        psi = psi_from_bins(ref_counts, cur_counts)
        js = jensen_shannon_divergence(ref_counts, cur_counts)
        ks = ks_statistic(pd.to_numeric(df_ref[col], errors="coerce"), pd.to_numeric(df_cur[col], errors="coerce"))

        sev = drift_severity(psi=psi, ks=ks, js=js, cv=None)

        rows.append({
            "feature": col,
            "type": "numeric",
            "psi": psi,
            "ks": ks,
            "js": js,
            "cramers_v": np.nan,
            "severity": sev
        })

    # Categorical drift
    for col in cat_cols:
        ref_map = profile["categorical"].get(col, {}).get("ref_counts", {})
        if not ref_map:
            continue

        ref_series = pd.Series(ref_map)
        cur_vc = df_cur[col].astype(str).fillna("MISSING").value_counts()

        # Alinear categorías
        all_cats = sorted(set(ref_series.index).union(set(cur_vc.index)))
        ref_counts = np.array([ref_series.get(c, 0) for c in all_cats], dtype=float)
        cur_counts = np.array([cur_vc.get(c, 0) for c in all_cats], dtype=float)

        psi = psi_from_bins(ref_counts, cur_counts)
        js = jensen_shannon_divergence(ref_counts, cur_counts)

        # Chi-square resumido como Cramér's V usando tabla 2xK (ref vs cur)
        ct = pd.DataFrame({"ref": ref_counts, "cur": cur_counts}, index=all_cats).T
        cv = cramers_v_from_crosstab(ct)

        sev = drift_severity(psi=psi, ks=None, js=js, cv=cv)

        rows.append({
            "feature": col,
            "type": "categorical",
            "psi": psi,
            "ks": np.nan,
            "js": js,
            "cramers_v": cv,
            "severity": sev
        })

    return pd.DataFrame(rows).sort_values(["severity", "psi"], ascending=[True, False])

# =========================
# Main de monitoreo
# =========================

def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    data_path = os.path.join(root, "Base_de_datos.csv")
    model_path = os.path.join(root, "best_model.joblib")
    profile_path = os.path.join(root, "baseline_profile.json")
    log_path = os.path.join(root, "monitoring_log.csv")
    latest_path = os.path.join(root, "latest_drift_metrics.csv")

    df = pd.read_csv(data_path)

    # Tipos básicos (si existe fecha)
    if "fecha_prestamo" in df.columns:
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")

    # Cargar modelo para generar pronóstico (si existe)
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        # Generar pronósticos
        if "Pago_atiempo" in df.columns:
            X_all = df.drop(columns=["Pago_atiempo"])
        else:
            X_all = df.copy()
        df["y_pred"] = model.predict(X_all)
    else:
        df["y_pred"] = np.nan

    # Definir baseline vs current (preferible por tiempo)
    if "fecha_prestamo" in df.columns and df["fecha_prestamo"].notna().any():
        df_sorted = df.sort_values("fecha_prestamo")
        cut = int(len(df_sorted) * 0.7)
        df_ref = df_sorted.iloc[:cut].copy()
        df_cur = df_sorted.iloc[cut:].copy()
        time_mode = "time_split_70_30"
    else:
        # fallback: split aleatorio (menos ideal, pero funciona)
        df_ref = df.sample(frac=0.7, random_state=42)
        df_cur = df.drop(df_ref.index)
        time_mode = "random_split_70_30"

    # Selección de columnas
    target = "Pago_atiempo"
    drop_cols = [c for c in [target, "y_pred"] if c in df.columns]

    X_ref = df_ref.drop(columns=drop_cols, errors="ignore")
    X_cur = df_cur.drop(columns=drop_cols, errors="ignore")

    numeric_cols = X_ref.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X_ref.select_dtypes(include=["object", "category"]).columns.tolist()

    # (Opcional) remover columnas que ya identificaste como leakage en modelado
    leakage_cols = [c for c in ["puntaje", "puntaje_datacredito"] if c in numeric_cols]
    for c in leakage_cols:
        numeric_cols.remove(c)

    # Crear baseline profile si no existe
    if not os.path.exists(profile_path):
        profile = build_baseline_profile(X_ref, numeric_cols=numeric_cols, cat_cols=cat_cols, n_bins=10)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile, f, ensure_ascii=False, indent=2)
        print(f"Baseline profile creado: {profile_path}")
    else:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

    # Drift global (ref vs cur)
    drift_df = compute_drift(X_ref, X_cur, profile, numeric_cols=numeric_cols, cat_cols=cat_cols)
    drift_df.to_csv(latest_path, index=False)
    print(f"Métricas drift guardadas: {latest_path}")

    # Monitoreo temporal (por mes) si hay fecha
    if "fecha_prestamo" in df.columns and df["fecha_prestamo"].notna().any():
        df_sorted = df.sort_values("fecha_prestamo")
        df_sorted["month"] = df_sorted["fecha_prestamo"].dt.to_period("M").astype(str)

        # Baseline fijo: primeros 6 meses o 70% (lo que aplique)
        base_months = df_sorted["month"].unique()[:6]
        df_base = df_sorted[df_sorted["month"].isin(base_months)]
        X_base = df_base.drop(columns=drop_cols + ["month"], errors="ignore")

        logs = []
        for m in df_sorted["month"].unique():
            df_m = df_sorted[df_sorted["month"] == m]
            X_m = df_m.drop(columns=drop_cols + ["month"], errors="ignore")

            drift_m = compute_drift(X_base, X_m, profile, numeric_cols=numeric_cols, cat_cols=cat_cols)
            # Resumen: promedio PSI y porcentaje rojo
            avg_psi = float(np.nanmean(drift_m["psi"].values))
            pct_red = float((drift_m["severity"] == "ROJO").mean())
            logs.append({"period": m, "avg_psi": avg_psi, "pct_red": pct_red, "mode": time_mode})

        log_df = pd.DataFrame(logs)
        # Append / overwrite
        log_df.to_csv(log_path, index=False)
        print(f"Log temporal guardado: {log_path}")

    # Recomendación automática
    red_feats = drift_df[drift_df["severity"] == "ROJO"]["feature"].tolist()
    if len(red_feats) > 0:
        print("\nALERTA: Drift crítico detectado en:", red_feats)
        print("Recomendación: revisar calidad de datos, umbrales y considerar reentrenamiento.")
    else:
        print("\nOK: No se detectó drift crítico (según umbrales actuales).")

if __name__ == "__main__":
    main()