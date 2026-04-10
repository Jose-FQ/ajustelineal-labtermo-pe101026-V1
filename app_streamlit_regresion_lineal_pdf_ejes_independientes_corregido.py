import io
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter
from scipy import stats

APP_DIR = Path(__file__).resolve().parent
UNAM_LOGO_PATH = APP_DIR / "Escudo-UNAM-escalable.png"
OWL_LOGO_PATH = APP_DIR / "Logo-Buho.png"

ACKNOWLEDGEMENT = (
    "Este trabajo se realizó con el apoyo del proyecto PAPIME, clave PE101026, "
    "titulado “Evaluación de las propiedades termodinámicas del dióxido de carbono "
    "y su relación con el calentamiento global: propuesta de tres protocolos de "
    "práctica experimental”."
)

# =========================
# Configuración inicial
# =========================

st.set_page_config(page_title="Ajuste lineal con PDF", page_icon="📈", layout="wide")


# =========================
# Utilidades de interfaz
# =========================

def render_header() -> None:
    col1, col2, col3 = st.columns([0.50, 2.8, 0.50])

    with col1:
        if UNAM_LOGO_PATH.exists():
            st.image(str(UNAM_LOGO_PATH), use_container_width=True)

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding-top:0.4rem;">
                <h1 style="margin-bottom:0.1rem;">AJUSTE LINEAL</h1>
                <h3 style="margin-top:0.1rem; margin-bottom:0.1rem; font-weight:500;">
                    Laboratorio de Termodinámica
                </h3>
                <h4 style="margin-top:0.1rem; margin-bottom:0.6rem; font-weight:400;">2026</h4>
                <div style="font-size:0.92rem; line-height:1.35; max-width:820px; margin:0 auto;">
                    {ACKNOWLEDGEMENT}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        if OWL_LOGO_PATH.exists():
            st.image(str(OWL_LOGO_PATH), use_container_width=True)

    st.divider()


def clean_var_name(name: str, default: str) -> str:
    name = (name or "").strip()
    return name if name else default


def clean_unit(unit: str) -> str:
    return (unit or "").strip()


def axis_label(name: str, unit: str) -> str:
    name = clean_var_name(name, "Variable")
    unit = clean_unit(unit)
    return f"{name} ({unit})" if unit else name


def slope_units(x_unit: str, y_unit: str) -> str:
    x_unit = clean_unit(x_unit)
    y_unit = clean_unit(y_unit)

    if x_unit and y_unit:
        return f"{y_unit}/{x_unit}"
    if y_unit:
        return f"{y_unit} por unidad de X"
    return "unidades de Y por unidad de X"


def format_number(value, mode: str = "Automático", sig_figs: int = 6) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    if np.isnan(value):
        return "NaN"
    if np.isposinf(value):
        return "∞"
    if np.isneginf(value):
        return "-∞"

    abs_v = abs(value)

    if mode == "Decimal":
        return f"{value:.{sig_figs}f}"
    if mode == "Científica":
        return f"{value:.{sig_figs}e}"

    if abs_v != 0 and (abs_v < 1e-3 or abs_v >= 1e4):
        return f"{value:.{sig_figs}e}"
    return f"{value:.{sig_figs}g}"


def display_value(value, mode: str, sig_figs: int, na_text: str = "No aplica") -> str:
    if value is None:
        return na_text
    if isinstance(value, float) and np.isnan(value):
        return na_text
    if isinstance(value, str):
        return value
    return format_number(value, mode, sig_figs)


def make_tick_formatter(mode: str = "Automático", sig_figs: int = 6):
    return FuncFormatter(lambda x, pos: format_number(x, mode, sig_figs))


def format_dataframe_for_display(df: pd.DataFrame, mode: str, sig_figs: int) -> pd.DataFrame:
    formatted = df.copy().astype(object)
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda v: format_number(v, mode, sig_figs))
    return formatted


def format_xy_dataframe_for_display(
    df: pd.DataFrame,
    x_mode: str,
    x_sig_figs: int,
    y_mode: str,
    y_sig_figs: int,
) -> pd.DataFrame:
    formatted = df.copy().astype(object)

    if formatted.shape[1] >= 1:
        col0 = formatted.columns[0]
        formatted[col0] = formatted[col0].map(
            lambda v: format_number(v, x_mode, x_sig_figs)
        )

    if formatted.shape[1] >= 2:
        col1 = formatted.columns[1]
        formatted[col1] = formatted[col1].map(
            lambda v: format_number(v, y_mode, y_sig_figs)
        )

    return formatted


def model_description(stats_dict: dict, results_mode: str, results_sig_figs: int) -> str:
    if stats_dict.get("intercept_fixed", False):
        return (
            "Intersección fija en "
            f"{format_number(stats_dict['fixed_intercept_value'], results_mode, results_sig_figs)}"
        )
    return "Intersección libre"


# =========================
# Lectura y preparación de datos
# =========================

def parse_text_data(text: str) -> pd.DataFrame:
    """
    Convierte texto pegado por el usuario en un DataFrame con columnas x, y.
    Acepta separadores: coma, punto y coma, tabulador o espacios.
    Ignora líneas vacías y líneas que comienzan con #.
    """
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        normalized = line.replace(";", ",").replace("\t", ",")

        if "," in normalized:
            parts = [p.strip() for p in normalized.split(",") if p.strip()]
        else:
            parts = [p.strip() for p in normalized.split() if p.strip()]

        if len(parts) < 2:
            raise ValueError(f"No se pudieron leer dos columnas en la línea: '{line}'")

        x_val = float(parts[0])
        y_val = float(parts[1])
        rows.append((x_val, y_val))

    if len(rows) < 2:
        raise ValueError("Se requieren al menos dos pares de datos (x, y).")

    return pd.DataFrame(rows, columns=["x", "y"])


def _score_candidate_dataframe(df: pd.DataFrame):
    if df is None or df.empty or df.shape[1] < 2:
        return -1, None

    lower_map = {str(c).strip().lower(): c for c in df.columns}

    if "x" in lower_map and "y" in lower_map:
        out = df[[lower_map["x"], lower_map["y"]]].copy()
        base_score = 1000
    else:
        out = df.iloc[:, :2].copy()
        base_score = 0

    out.columns = ["x", "y"]
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna().reset_index(drop=True)

    score = base_score + len(out)
    return score, out


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """
    Lee un CSV o TXT subido por el usuario.
    Si existen columnas llamadas x e y, las usa.
    En caso contrario, toma las primeras dos columnas válidas.
    Intenta archivos con o sin encabezado.
    """
    raw_bytes = uploaded_file.getvalue()
    try:
        content = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        content = raw_bytes.decode("latin-1")

    candidates = []

    common_attempts = [
        {"sep": ",", "engine": "python", "comment": "#"},
        {"sep": r"[,;\t\s]+", "engine": "python", "comment": "#"},
        {"sep": None, "engine": "python", "comment": "#"},
    ]

    for kwargs in common_attempts:
        try:
            df = pd.read_csv(io.StringIO(content), **kwargs)
            candidates.append(df)
        except Exception:
            pass

    for kwargs in common_attempts:
        try:
            df = pd.read_csv(io.StringIO(content), header=None, **kwargs)
            candidates.append(df)
        except Exception:
            pass

    best_score = -1
    best_df = None

    for candidate in candidates:
        score, out = _score_candidate_dataframe(candidate)
        if score > best_score and out is not None:
            best_score = score
            best_df = out

    if best_df is None or len(best_df) < 2:
        raise ValueError(
            "No fue posible interpretar el archivo. Verifique que contenga al menos "
            "dos columnas con datos numéricos."
        )

    return best_df


# =========================
# Cálculo estadístico
# =========================

def safe_pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])



def linear_regression_analysis(
    df: pd.DataFrame,
    intercept_mode: str = "libre",
    fixed_intercept: float = 0.0,
) -> dict:
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    n = len(df)

    if n < 3:
        raise ValueError(
            "Se requieren al menos 3 puntos para reportar estadísticos del ajuste "
            "con mayor sentido."
        )

    y_mean = np.mean(y)
    centered_sst = np.sum((y - y_mean) ** 2)
    alpha = 0.05

    if intercept_mode == "fija":
        intercept = float(fixed_intercept)
        sxx_fixed = np.sum(x ** 2)

        if np.isclose(sxx_fixed, 0.0):
            raise ValueError(
                "No es posible estimar la pendiente con intersección fija porque "
                "la suma de x² es cero."
            )

        slope = np.sum(x * (y - intercept)) / sxx_fixed
        y_pred = intercept + slope * x
        residuals = y - y_pred

        dof = n - 1
        sse = np.sum(residuals ** 2)
        mse = sse / dof
        rmse = np.sqrt(mse)

        slope_stderr = np.sqrt(mse / sxx_fixed)
        if np.isclose(slope_stderr, 0.0):
            if np.isclose(slope, 0.0):
                t_stat_slope = 0.0
                p_value = 1.0
            else:
                t_stat_slope = np.inf
                p_value = 0.0
        else:
            t_stat_slope = slope / slope_stderr
            p_value = 2 * stats.t.sf(abs(t_stat_slope), dof)

        t_crit = stats.t.ppf(1 - alpha / 2, dof)
        slope_ci = (slope - t_crit * slope_stderr, slope + t_crit * slope_stderr)

        if centered_sst > 0:
            r2 = 1 - (sse / centered_sst)
        else:
            r2 = 1.0 if np.isclose(sse, 0.0) else np.nan

        r_value = safe_pearson_r(x, y)
        f_stat = t_stat_slope ** 2 if np.isfinite(t_stat_slope) else np.inf

        return {
            "n": n,
            "x": x,
            "y": y,
            "slope": slope,
            "intercept": intercept,
            "r": r_value,
            "r2": r2,
            "p_value": p_value,
            "slope_stderr": slope_stderr,
            "intercept_stderr": None,
            "slope_ci": slope_ci,
            "intercept_ci": None,
            "y_pred": y_pred,
            "residuals": residuals,
            "sse": sse,
            "mse": mse,
            "rmse": rmse,
            "f_stat": f_stat,
            "dof": dof,
            "x_mean": np.mean(x),
            "sxx": sxx_fixed,
            "intercept_fixed": True,
            "fixed_intercept_value": intercept,
            "t_stat_slope": t_stat_slope,
            "r2_note": (
                "R² se calculó como 1 - SSE/SST usando SST centrada respecto a la media de Y. "
                "Cuando se impone una intersección fija, R² puede tomar valores negativos."
            ),
        }

    if np.allclose(x, x[0]):
        raise ValueError(
            "Todos los valores de X son iguales. No es posible calcular una recta "
            "de regresión con pendiente definida."
        )

    result = stats.linregress(x, y)

    slope = result.slope
    intercept = result.intercept
    r_value = result.rvalue
    r2 = r_value ** 2
    p_value = result.pvalue
    slope_stderr = result.stderr
    intercept_stderr = result.intercept_stderr

    y_pred = intercept + slope * x
    residuals = y - y_pred

    dof = n - 2
    sse = np.sum(residuals ** 2)
    mse = sse / dof
    rmse = np.sqrt(mse)

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    ss_reg = np.sum((y_pred - y_mean) ** 2)
    f_stat = (ss_reg / 1) / (sse / dof) if sse > 0 else np.inf

    t_crit = stats.t.ppf(1 - alpha / 2, dof)

    slope_ci = (slope - t_crit * slope_stderr, slope + t_crit * slope_stderr)
    intercept_ci = (
        intercept - t_crit * intercept_stderr,
        intercept + t_crit * intercept_stderr,
    )

    return {
        "n": n,
        "x": x,
        "y": y,
        "slope": slope,
        "intercept": intercept,
        "r": r_value,
        "r2": r2,
        "p_value": p_value,
        "slope_stderr": slope_stderr,
        "intercept_stderr": intercept_stderr,
        "slope_ci": slope_ci,
        "intercept_ci": intercept_ci,
        "y_pred": y_pred,
        "residuals": residuals,
        "sse": sse,
        "mse": mse,
        "rmse": rmse,
        "f_stat": f_stat,
        "dof": dof,
        "x_mean": x_mean,
        "sxx": sxx,
        "intercept_fixed": False,
        "fixed_intercept_value": None,
        "t_stat_slope": result.slope / result.stderr if result.stderr != 0 else np.inf,
        "r2_note": "R² = r² para el ajuste lineal con intersección libre.",
    }


# =========================
# Gráficas
# =========================

def build_main_figure(
    x,
    y,
    y_pred,
    slope,
    intercept,
    r2,
    x_name,
    x_unit,
    y_name,
    y_unit,
    x_axis_mode,
    x_axis_sig_figs,
    y_axis_mode,
    y_axis_sig_figs,
    results_mode,
    results_sig_figs,
    intercept_fixed=False,
):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = np.argsort(x)

    x_label = axis_label(x_name, x_unit)
    y_label = axis_label(y_name, y_unit)

    ax.scatter(x, y, label="Datos experimentales")
    ax.plot(x[order], y_pred[order], label="Ajuste lineal")
    ax.set_title("Datos y ajuste lineal")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.xaxis.set_major_formatter(make_tick_formatter(x_axis_mode, x_axis_sig_figs))
    ax.yaxis.set_major_formatter(make_tick_formatter(y_axis_mode, y_axis_sig_figs))

    eq_lines = [
        f"{clean_var_name(y_name, 'Y')} = "
        f"{format_number(intercept, results_mode, results_sig_figs)} + "
        f"{format_number(slope, results_mode, results_sig_figs)}·"
        f"{clean_var_name(x_name, 'X')}",
        f"R² = {format_number(r2, 'Decimal', 6)}",
    ]

    if intercept_fixed:
        eq_lines.append("Intersección fija")

    ax.text(
        0.03,
        0.97,
        "\n".join(eq_lines),
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()
    return fig



def build_residual_figure(y_pred, residuals, y_axis_mode, y_axis_sig_figs):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(y_pred, residuals)
    ax.axhline(0, linestyle="--")
    ax.set_title("Gráfica de residuales")
    ax.set_xlabel("Valores ajustados")
    ax.set_ylabel("Residuales")
    ax.grid(True, alpha=0.3)

    # Ambos ejes están en unidades de Y
    ax.xaxis.set_major_formatter(make_tick_formatter(y_axis_mode, y_axis_sig_figs))
    ax.yaxis.set_major_formatter(make_tick_formatter(y_axis_mode, y_axis_sig_figs))

    fig.tight_layout()
    return fig


# =========================
# Interpretación y reporte
# =========================

def build_interpretation_markdown(
    stats_dict: dict,
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    results_mode: str,
    results_sig_figs: int,
) -> str:
    x_name = clean_var_name(x_name, "X")
    y_name = clean_var_name(y_name, "Y")
    x_unit = clean_unit(x_unit)
    y_unit = clean_unit(y_unit)

    slope_unit_text = slope_units(x_unit, y_unit)
    intercept_unit_text = y_unit if y_unit else "mismas unidades de Y"

    if stats_dict["p_value"] < 0.05:
        significance_text = (
            "La pendiente es estadísticamente distinta de cero al 95% de confianza."
        )
    else:
        significance_text = (
            "La pendiente no es estadísticamente distinta de cero al 95% de confianza."
        )

    r2_value = stats_dict["r2"]
    if not np.isnan(r2_value):
        if r2_value >= 0.90:
            r2_text = (
                f"El modelo lineal explica una proporción muy alta de la variabilidad de {y_name}."
            )
        elif r2_value >= 0.70:
            r2_text = (
                f"El modelo lineal explica una proporción alta de la variabilidad de {y_name}."
            )
        elif r2_value >= 0.50:
            r2_text = (
                f"El modelo lineal explica una proporción moderada de la variabilidad de {y_name}."
            )
        elif r2_value >= 0.0:
            r2_text = (
                f"El modelo lineal explica una proporción limitada de la variabilidad de {y_name}."
            )
        else:
            r2_text = (
                "El valor de R² es negativo. Esto puede ocurrir cuando se impone una "
                "intersección fija y el modelo ajusta peor que usar simplemente la media de Y."
            )
    else:
        r2_text = "No fue posible calcular un R² interpretable con estos datos."

    if stats_dict.get("intercept_fixed", False):
        intercept_section = f"""
**Intersección (b0)**  
La intersección no fue estimada a partir de los datos: se fijó manualmente en **{format_number(stats_dict['fixed_intercept_value'], results_mode, results_sig_figs)}**.  
Unidades de la intersección: **{intercept_unit_text}**  
Por lo tanto, no se reportan error estándar ni intervalo de confianza para la intersección.
"""
        uncertainty_section = f"""
**Error estándar de la pendiente**  
EE(b1) = **{format_number(stats_dict["slope_stderr"], results_mode, results_sig_figs)}**

**Intervalo de confianza al 95% para la pendiente**  
IC95% pendiente: **[{format_number(stats_dict["slope_ci"][0], results_mode, results_sig_figs)}, {format_number(stats_dict["slope_ci"][1], results_mode, results_sig_figs)}]**

**Error estándar e intervalo de confianza de la intersección**  
No aplica, porque la intersección fue fijada por el usuario y no fue estimada por regresión.
"""
    else:
        intercept_section = f"""
**Intersección (b0)**  
Representa el valor estimado de **{y_name}** cuando **{x_name} = 0**.  
Valor estimado: **{format_number(stats_dict["intercept"], results_mode, results_sig_figs)}**  
Unidades de la intersección: **{intercept_unit_text}**  
Esta interpretación solo tiene sentido físico si el valor **{x_name} = 0** es accesible y relevante en el sistema experimental.
"""
        uncertainty_section = f"""
**Error estándar de la pendiente e intersección**  
Miden la incertidumbre estadística asociada a cada parámetro del ajuste.
- EE(b1): **{format_number(stats_dict["slope_stderr"], results_mode, results_sig_figs)}**
- EE(b0): **{format_number(stats_dict["intercept_stderr"], results_mode, results_sig_figs)}**

**Intervalos de confianza al 95%**  
Indican el intervalo plausible donde puede encontrarse el valor real del parámetro:
- IC95% pendiente: **[{format_number(stats_dict["slope_ci"][0], results_mode, results_sig_figs)}, {format_number(stats_dict["slope_ci"][1], results_mode, results_sig_figs)}]**
- IC95% intersección: **[{format_number(stats_dict["intercept_ci"][0], results_mode, results_sig_figs)}, {format_number(stats_dict["intercept_ci"][1], results_mode, results_sig_figs)}]**
"""

    return f"""
**Modelo ajustado**  
{y_name} = b0 + b1·{x_name}

**Configuración del modelo**  
{model_description(stats_dict, results_mode, results_sig_figs)}

**Pendiente (b1)**  
Representa el cambio promedio de **{y_name}** cuando **{x_name}** aumenta una unidad.  
Valor estimado: **{format_number(stats_dict["slope"], results_mode, results_sig_figs)}**  
Unidades de la pendiente: **{slope_unit_text}**

{intercept_section}
{uncertainty_section}

**Coeficiente de determinación R²**  
Valor: **{format_number(stats_dict["r2"], "Decimal", 6)}**  
{r2_text}  
{stats_dict['r2_note']}

**RMSE**  
Valor: **{format_number(stats_dict["rmse"], results_mode, results_sig_figs)}**  
Representa el tamaño típico del error del ajuste en unidades de **{y_name}**.

**p(pendiente)**  
Valor: **{format_number(stats_dict["p_value"], "Científica", results_sig_figs)}**  
{significance_text}  
El valor p(pendiente) sirve para evaluar si hay evidencia estadística de que la pendiente verdadera es distinta de cero, bajo la configuración elegida para la intersección.

**Residuales**  
Son las diferencias entre los valores experimentales y los valores calculados por el modelo.  
Se recomienda revisar la gráfica de residuales para detectar curvatura, heterocedasticidad o valores atípicos.  

*Heterocedasticidad: la dispersión de errores cambia con x o con y.*
"""



def analysis_text(
    stats_dict: dict,
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    results_mode: str,
    results_sig_figs: int,
) -> str:
    x_name = clean_var_name(x_name, "X")
    y_name = clean_var_name(y_name, "Y")

    if stats_dict["p_value"] < 0.05:
        significance_text = (
            "La pendiente es estadísticamente significativa a un nivel de confianza del 95%."
        )
    else:
        significance_text = (
            "La pendiente no resulta estadísticamente significativa a un nivel de confianza del 95%."
        )

    r2_value = stats_dict["r2"]
    if not np.isnan(r2_value):
        if r2_value >= 0.90:
            r2_text = f"El ajuste lineal explica una proporción muy alta de la variabilidad de {y_name}."
        elif r2_value >= 0.70:
            r2_text = f"El ajuste lineal explica una proporción alta de la variabilidad de {y_name}."
        elif r2_value >= 0.50:
            r2_text = f"El ajuste lineal explica una proporción moderada de la variabilidad de {y_name}."
        elif r2_value >= 0.0:
            r2_text = f"El ajuste lineal explica una proporción limitada de la variabilidad de {y_name}."
        else:
            r2_text = (
                "R² es negativo; esto puede ocurrir cuando la intersección se fija y el modelo "
                "ajusta peor que la media de Y."
            )
    else:
        r2_text = "No fue posible calcular un R² interpretable con estos datos."

    if stats_dict.get("intercept_fixed", False):
        intercept_interpretation = (
            "La intersección no se estimó: se fijó manualmente por el usuario. "
            "No aplica reportar su error estándar ni su intervalo de confianza."
        )
        intercept_stderr_text = "No aplica (valor fijo)"
        intercept_ci_text = "No aplica (valor fijo)"
    else:
        intercept_interpretation = (
            f"La intersección representa el valor estimado de {y_name} cuando {x_name} = 0."
        )
        intercept_stderr_text = format_number(
            stats_dict["intercept_stderr"], results_mode, results_sig_figs
        )
        intercept_ci_text = (
            f"[{format_number(stats_dict['intercept_ci'][0], results_mode, results_sig_figs)}, "
            f"{format_number(stats_dict['intercept_ci'][1], results_mode, results_sig_figs)}]"
        )

    interpretation_lines = [
        significance_text,
        r2_text,
        f"La pendiente representa el cambio promedio de {y_name} por cada unidad de {x_name}.",
        intercept_interpretation,
        "Revise la gráfica de residuales para detectar curvatura, heterocedasticidad o valores atípicos.",
    ]

    wrapped_interpretation = "\n".join(
        f"    - {textwrap.fill(line, width=88, subsequent_indent='      ')}"
        for line in interpretation_lines
    )

    return f"""
REPORTE DE AJUSTE LINEAL

Modelo:
    {y_name} = b0 + b1*{x_name}

Configuración del modelo:
    {model_description(stats_dict, results_mode, results_sig_figs)}

Etiquetas de variables:
    X = {axis_label(x_name, x_unit)}
    Y = {axis_label(y_name, y_unit)}

Parámetros estimados:
    Intersección (b0) = {format_number(stats_dict["intercept"], results_mode, results_sig_figs)}
    Pendiente   (b1) = {format_number(stats_dict["slope"], results_mode, results_sig_figs)}

Interpretación física básica:
    La pendiente tiene unidades de {slope_units(x_unit, y_unit)}.
    La intersección tiene unidades de {clean_unit(y_unit) if clean_unit(y_unit) else 'Y'}.

Errores estándar:
    EE(b0) = {intercept_stderr_text}
    EE(b1) = {format_number(stats_dict["slope_stderr"], results_mode, results_sig_figs)}

Intervalos de confianza al 95%:
    IC95%(b0) = {intercept_ci_text}
    IC95%(b1) = [{format_number(stats_dict["slope_ci"][0], results_mode, results_sig_figs)}, {format_number(stats_dict["slope_ci"][1], results_mode, results_sig_figs)}]

Pruebas y bondad de ajuste:
    n             = {stats_dict["n"]}
    gl            = {stats_dict["dof"]}
    r             = {display_value(stats_dict["r"], results_mode, results_sig_figs)}
    R²            = {display_value(stats_dict["r2"], 'Decimal', 6)}
    p(pendiente)  = {format_number(stats_dict["p_value"], 'Científica', results_sig_figs)}
    SSE           = {format_number(stats_dict["sse"], results_mode, results_sig_figs)}
    MSE           = {format_number(stats_dict["mse"], results_mode, results_sig_figs)}
    RMSE          = {format_number(stats_dict["rmse"], results_mode, results_sig_figs)}
    F             = {display_value(stats_dict["f_stat"], results_mode, results_sig_figs)}

Nota sobre R²:
    {stats_dict['r2_note']}

Interpretación breve:
{wrapped_interpretation}
""".strip()



def create_pdf_bytes(
    df: pd.DataFrame,
    stats_dict: dict,
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    x_axis_mode: str,
    x_axis_sig_figs: int,
    y_axis_mode: str,
    y_axis_sig_figs: int,
    results_mode: str,
    results_sig_figs: int,
) -> bytes:
    buffer = io.BytesIO()

    with PdfPages(buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = "Reporte de ajuste lineal"
        metadata["Author"] = "App Streamlit"
        metadata["Subject"] = "Regresión lineal simple"
        metadata["CreationDate"] = datetime.now()

        fig1 = build_main_figure(
            stats_dict["x"],
            stats_dict["y"],
            stats_dict["y_pred"],
            stats_dict["slope"],
            stats_dict["intercept"],
            stats_dict["r2"],
            x_name,
            x_unit,
            y_name,
            y_unit,
            x_axis_mode,
            x_axis_sig_figs,
            y_axis_mode,
            y_axis_sig_figs,
            results_mode,
            results_sig_figs,
            intercept_fixed=stats_dict.get("intercept_fixed", False),
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        fig2 = build_residual_figure(
            stats_dict["y_pred"],
            stats_dict["residuals"],
            y_axis_mode,
            y_axis_sig_figs,
        )
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(8.5, 11))
        ax3.axis("off")
        ax3.text(
            0.03,
            0.97,
            analysis_text(
                stats_dict,
                x_name,
                x_unit,
                y_name,
                y_unit,
                results_mode,
                results_sig_figs,
            ),
            va="top",
            ha="left",
            family="monospace",
            fontsize=9.5,
        )
        fig3.tight_layout()
        pdf.savefig(fig3, bbox_inches="tight")
        plt.close(fig3)

        fig4, ax4 = plt.subplots(figsize=(8.5, 11))
        ax4.axis("off")
        ax4.set_title("Datos experimentales", fontsize=14, pad=14)

        table_df = df.copy()
        table_df.columns = [
            axis_label(x_name, x_unit),
            axis_label(y_name, y_unit),
        ]
        table_df = format_xy_dataframe_for_display(
            table_df,
            x_axis_mode,
            x_axis_sig_figs,
            y_axis_mode,
            y_axis_sig_figs,
        )
        table_df.index = np.arange(1, len(table_df) + 1)

        table = ax4.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            rowLabels=table_df.index,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.35)
        fig4.tight_layout()
        pdf.savefig(fig4, bbox_inches="tight")
        plt.close(fig4)

    buffer.seek(0)
    return buffer.getvalue()


# =========================
# Interfaz principal
# =========================

render_header()

st.title("📈 Ajuste lineal de datos experimentales")
st.write(
    "Esta aplicación realiza un ajuste lineal simple de datos experimentales, "
    "permite definir qué variable se coloca en cada eje, mostrar formatos "
    "numéricos distintos para el eje X y el eje Y, elegir si la ordenada al origen "
    "se estima libremente o se fija en cero o en cualquier otro valor, interpretar "
    "el significado físico de los parámetros y generar un reporte en PDF."
)

with st.expander("Instrucciones de uso", expanded=True):
    st.markdown(
        """
### Pasos de uso
1. Escriba el nombre de la variable independiente y, si lo desea, su unidad.
2. Escriba el nombre de la variable dependiente y, si lo desea, su unidad.
3. Elija si la ordenada al origen será libre o fija. Si será fija, capture el valor deseado.
4. Pegue los datos en dos columnas o cargue un archivo CSV/TXT.
5. Seleccione el formato de resultados y el formato específico para cada eje.
6. Presione **Calcular ajuste**.
7. Revise la ecuación ajustada, la gráfica, los residuales, el resumen estadístico y la interpretación física.
8. Descargue el reporte PDF.

### Formatos válidos para ingresar datos
- Separados por coma: `1, 2.1`
- Separados por espacio: `1 2.1`
- Separados por tabulador
- Notación científica: `1.0e-6, 2.5e-4`

### Recomendaciones
- Use al menos 3 puntos experimentales.
- Verifique que una relación lineal tenga sentido físico para el sistema estudiado.
- Si fija la ordenada al origen, hágalo solo cuando exista una justificación experimental o teórica.
- Revise la gráfica de residuales para identificar posibles desviaciones del comportamiento lineal.
- Para indicar los ejes puede usar el formato LaTeX por ejemplo $\Delta T$
        """
    )

with st.expander("Ver formato de entrada manual"):
    st.code(
        """1, 2.1
2, 4.0
3, 5.9
4, 8.2
5, 10.1
6, 1.22e1
7, 1.38e1
8, 1.61e1""",
        language="text",
    )

st.subheader("Definición de variables")
v1, v2 = st.columns(2)

with v1:
    x_name = st.text_input("Nombre de la variable del eje X", value="x")
    x_unit = st.text_input("Unidad de X", value="")

with v2:
    y_name = st.text_input("Nombre de la variable del eje Y", value="y")
    y_unit = st.text_input("Unidad de Y", value="")

st.subheader("Configuración del modelo")
mc1, mc2 = st.columns([1.5, 1])

with mc1:
    intercept_mode_label = st.radio(
        "Tratamiento de la ordenada al origen",
        options=["Libre (estimarla con los datos)", "Fija"],
        horizontal=True,
    )

with mc2:
    fixed_intercept_value = st.number_input(
        "Valor fijo de la ordenada al origen",
        value=0.0,
        format="%.10g",
        disabled=intercept_mode_label != "Fija",
        help="Puede fijarla en 0 o en cualquier otro valor físicamente justificado.",
    )

intercept_mode = "fija" if intercept_mode_label == "Fija" else "libre"

st.subheader("Formato numérico")

fr1, fr2 = st.columns([1.3, 1])
with fr1:
    results_mode = st.selectbox(
        "Formato de números para resultados y parámetros",
        ["Automático", "Decimal", "Científica"],
        index=0,
    )
with fr2:
    results_sig_figs = st.slider(
        "Cifras significativas para resultados",
        min_value=2,
        max_value=10,
        value=6,
    )

fx1, fx2, fy1, fy2 = st.columns([1.2, 1, 1.2, 1])

with fx1:
    x_axis_mode = st.selectbox(
        "Formato del eje X",
        ["Automático", "Decimal", "Científica"],
        index=0,
    )
with fx2:
    x_axis_sig_figs = st.slider(
        "Cifras del eje X",
        min_value=2,
        max_value=10,
        value=6,
        key="x_axis_sig_figs",
    )
with fy1:
    y_axis_mode = st.selectbox(
        "Formato del eje Y",
        ["Automático", "Decimal", "Científica"],
        index=0,
    )
with fy2:
    y_axis_sig_figs = st.slider(
        "Cifras del eje Y",
        min_value=2,
        max_value=10,
        value=6,
        key="y_axis_sig_figs",
    )

col1, col2 = st.columns([1.2, 1])

with col1:
    input_mode = st.radio(
        "Método de entrada",
        options=["Pegar datos", "Subir archivo CSV/TXT"],
        horizontal=True,
    )

    if input_mode == "Pegar datos":
        raw_text = st.text_area(
            "Pegue aquí los datos X,Y (una fila por punto)",
            height=220,
            value="1, 2.1\n2, 4.0\n3, 5.9\n4, 8.2\n5, 10.1\n6, 12.2\n7, 13.8\n8, 16.1",
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Suba un archivo CSV o TXT",
            type=["csv", "txt"],
            help=(
                "Puede usar columnas llamadas x e y, o simplemente dejar la variable X "
                "en la primera columna y la variable Y en la segunda."
            ),
        )
        raw_text = None

    calcular = st.button("Calcular ajuste", type="primary")

with col2:
    st.subheader("Qué incluye el reporte")
    st.markdown(
        """
- Ecuación de la recta ajustada
- Parámetros estadísticos básicos
- Intervalos de confianza al 95%
- Gráfica de datos con ajuste
- Gráfica de residuales
- Tabla de datos experimentales
- Interpretación física de los parámetros
        """
    )

    st.subheader("Etiquetas y formatos seleccionados")
    st.markdown(
        f"""
- **Eje X:** {axis_label(x_name, x_unit)} — {x_axis_mode}, {x_axis_sig_figs} cifras
- **Eje Y:** {axis_label(y_name, y_unit)} — {y_axis_mode}, {y_axis_sig_figs} cifras
- **Resultados:** {results_mode}, {results_sig_figs} cifras
- **Modelo:** {('Intersección fija en ' + format_number(fixed_intercept_value, results_mode, results_sig_figs)) if intercept_mode == 'fija' else 'Intersección libre'}
- **Unidades de la pendiente:** {slope_units(x_unit, y_unit)}
        """
    )

if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "stats_dict" not in st.session_state:
    st.session_state.stats_dict = None
if "df_data" not in st.session_state:
    st.session_state.df_data = None

if calcular:
    try:
        if input_mode == "Pegar datos":
            df = parse_text_data(raw_text)
        else:
            if uploaded_file is None:
                raise ValueError("Debe subir un archivo antes de calcular.")
            df = read_uploaded_file(uploaded_file)

        if len(df) < 3:
            raise ValueError("Se requieren al menos 3 puntos válidos.")

        stats_dict = linear_regression_analysis(
            df,
            intercept_mode=intercept_mode,
            fixed_intercept=fixed_intercept_value,
        )
        pdf_bytes = create_pdf_bytes(
            df,
            stats_dict,
            x_name,
            x_unit,
            y_name,
            y_unit,
            x_axis_mode,
            x_axis_sig_figs,
            y_axis_mode,
            y_axis_sig_figs,
            results_mode,
            results_sig_figs,
        )

        st.session_state.df_data = df
        st.session_state.stats_dict = stats_dict
        st.session_state.pdf_bytes = pdf_bytes

        st.success(
            "Ajuste calculado correctamente. Ya puede revisar resultados y descargar el PDF."
        )

    except Exception as e:
        st.session_state.df_data = None
        st.session_state.stats_dict = None
        st.session_state.pdf_bytes = None
        st.error(f"Ocurrió un error: {e}")

if st.session_state.df_data is not None and st.session_state.stats_dict is not None:
    df = st.session_state.df_data
    stats_dict = st.session_state.stats_dict

    st.session_state.pdf_bytes = create_pdf_bytes(
        df,
        stats_dict,
        x_name,
        x_unit,
        y_name,
        y_unit,
        x_axis_mode,
        x_axis_sig_figs,
        y_axis_mode,
        y_axis_sig_figs,
        results_mode,
        results_sig_figs,
    )

    st.divider()
    st.subheader("Datos leídos")

    display_df = df.copy()
    display_df.columns = [
        axis_label(x_name, x_unit),
        axis_label(y_name, y_unit),
    ]
    st.dataframe(
        format_xy_dataframe_for_display(
            display_df,
            x_axis_mode,
            x_axis_sig_figs,
            y_axis_mode,
            y_axis_sig_figs,
        ),
        use_container_width=True,
    )

    st.info(f"Modelo calculado: {model_description(stats_dict, results_mode, results_sig_figs)}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pendiente", format_number(stats_dict["slope"], results_mode, results_sig_figs))
    m2.metric(
        "Error de pendiente",
        format_number(stats_dict["slope_stderr"], results_mode, results_sig_figs),
    )
    m3.metric(
        "Intersección",
        format_number(stats_dict["intercept"], results_mode, results_sig_figs),
    )
    m4.metric(
        "Error de intersección",
        display_value(stats_dict["intercept_stderr"], results_mode, results_sig_figs),
    )
    m5.metric("R²", display_value(stats_dict["r2"], "Decimal", 6))

    g1, g2 = st.columns(2)

    with g1:
        fig_main = build_main_figure(
            stats_dict["x"],
            stats_dict["y"],
            stats_dict["y_pred"],
            stats_dict["slope"],
            stats_dict["intercept"],
            stats_dict["r2"],
            x_name,
            x_unit,
            y_name,
            y_unit,
            x_axis_mode,
            x_axis_sig_figs,
            y_axis_mode,
            y_axis_sig_figs,
            results_mode,
            results_sig_figs,
            intercept_fixed=stats_dict.get("intercept_fixed", False),
        )
        st.pyplot(fig_main)
        plt.close(fig_main)

    with g2:
        fig_res = build_residual_figure(
            stats_dict["y_pred"],
            stats_dict["residuals"],
            y_axis_mode,
            y_axis_sig_figs,
        )
        st.pyplot(fig_res)
        plt.close(fig_res)

    st.subheader("Resumen estadístico")
    summary_df = pd.DataFrame(
        {
            "Parámetro": [
                "Configuración del modelo",
                "n",
                "Pendiente",
                "Intersección",
                "r",
                "R²",
                "Error estándar de pendiente",
                "Error estándar de la intersección",
                "IC95% pendiente (inferior)",
                "IC95% pendiente (superior)",
                "IC95% intersección (inferior)",
                "IC95% intersección (superior)",
                "SSE",
                "MSE",
                "RMSE",
                "Estadístico F del modelo lineal",
                "p(pendiente)",
            ],
            "Valor": [
                model_description(stats_dict, results_mode, results_sig_figs),
                format_number(stats_dict["n"], results_mode, results_sig_figs),
                format_number(stats_dict["slope"], results_mode, results_sig_figs),
                format_number(stats_dict["intercept"], results_mode, results_sig_figs),
                display_value(stats_dict["r"], results_mode, results_sig_figs),
                display_value(stats_dict["r2"], "Decimal", 6),
                format_number(stats_dict["slope_stderr"], results_mode, results_sig_figs),
                display_value(stats_dict["intercept_stderr"], results_mode, results_sig_figs),
                format_number(stats_dict["slope_ci"][0], results_mode, results_sig_figs),
                format_number(stats_dict["slope_ci"][1], results_mode, results_sig_figs),
                display_value(
                    None if stats_dict["intercept_ci"] is None else stats_dict["intercept_ci"][0],
                    results_mode,
                    results_sig_figs,
                ),
                display_value(
                    None if stats_dict["intercept_ci"] is None else stats_dict["intercept_ci"][1],
                    results_mode,
                    results_sig_figs,
                ),
                format_number(stats_dict["sse"], results_mode, results_sig_figs),
                format_number(stats_dict["mse"], results_mode, results_sig_figs),
                format_number(stats_dict["rmse"], results_mode, results_sig_figs),
                display_value(stats_dict["f_stat"], results_mode, results_sig_figs),
                format_number(stats_dict["p_value"], "Científica", results_sig_figs),
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True)

    if stats_dict.get("intercept_fixed", False):
        st.caption(stats_dict["r2_note"])

    st.subheader("Interpretación física de los resultados")
    with st.expander("Ver interpretación", expanded=True):
        st.markdown(
            build_interpretation_markdown(
                stats_dict,
                x_name,
                x_unit,
                y_name,
                y_unit,
                results_mode,
                results_sig_figs,
            )
        )

if st.session_state.pdf_bytes is not None:
    st.divider()
    st.download_button(
        label="Descargar reporte PDF",
        data=st.session_state.pdf_bytes,
        file_name="reporte_ajuste_lineal.pdf",
        mime="application/pdf",
    )

st.divider()
st.caption(
    "Estimado usuario o estimada usuaria: si considera que alguna parte de la aplicación "
    "puede mejorarse, por favor hágamelo saber al correo jllopezcervantes@quimica.unam.mx. "
    "Haremos todo lo posible por atender sus comentarios y mejorarla. "
    "Atentamente, José Luis López Cervantes."
)
