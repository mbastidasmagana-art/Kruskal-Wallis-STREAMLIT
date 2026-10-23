# ============================================================
# App: Prueba de Hipótesis de Kruskal-Wallis
# Autor: Martin, Andres
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Configuración de la página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Prueba de Kruskal-Wallis",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Prueba de Hipótesis de Kruskal-Wallis")
st.markdown(
    "Prueba **no paramétrica** para comparar si **tres o más grupos independientes** "
    "provienen de la misma distribución."
    "Autores: Martin Bastidas y Andres Erazo"
)
st.markdown("Autores: Martin Bastidas y Andres Erazo")

st.divider()

# ------------------------------------------------------------
# 1. Parámetros de entrada
# ------------------------------------------------------------
st.header("1️⃣ Configuración de la prueba")

col1, col2 = st.columns(2)
with col1:
    k = st.number_input(
        "Número de grupos (k)",
        min_value=3, max_value=10, value=3, step=1
    )
with col2:
    alpha = st.selectbox(
        "Nivel de significancia (α)",
        options=[0.01, 0.05, 0.10],
        index=1
    )

# Limpia claves obsoletas si se reduce k (error estructural #4)
prev_k = st.session_state.get("prev_k", k)
if prev_k > k:
    for i in range(k + 1, prev_k + 1):
        st.session_state.pop(f"grupo_{i}", None)
st.session_state["prev_k"] = k

st.divider()

# ------------------------------------------------------------
# 2. Ingreso de datos por grupo
# ------------------------------------------------------------
st.header("2️⃣ Ingreso de datos")
st.caption("Ingresa los valores de cada grupo separados por **comas** o **espacios**.")

grupos = {}
nombres_grupos = []
datos_grupos = []
errores = []

for i in range(1, k + 1):
    texto = st.text_input(
        f"Grupo {i}",
        placeholder="Ej: 12, 15, 18, 20, 22",
        key=f"grupo_{i}"
    )
    nombre = f"Grupo {i}"
    if not texto.strip():
        errores.append(f"El {nombre} está vacío.")
        continue
    try:
        valores = [float(x) for x in texto.replace(",", " ").split()]
    except ValueError:
        st.error(f"❌ El {nombre} contiene valores no numéricos.")
        errores.append(f"{nombre} no numérico.")
        continue
    if len(valores) < 2:
        st.warning(f"⚠️ El {nombre} debe tener al menos 2 observaciones.")
        errores.append(f"{nombre} con <2 observaciones.")
        continue
    grupos[nombre] = valores
    nombres_grupos.append(nombre)
    datos_grupos.append(np.asarray(valores, dtype=float))

entradas_validas = (len(errores) == 0) and (len(grupos) == k)

st.divider()


# ------------------------------------------------------------
# Funciones auxiliares
# ------------------------------------------------------------
def calcular_kw(datos_grupos):
    """H sin corrección, factor C, H corregido, p-valor, rangos por grupo."""
    N = sum(len(g) for g in datos_grupos)
    todos = np.concatenate(datos_grupos)
    rangos = stats.rankdata(todos)

    suma_rangos = []
    prom_rangos = []
    idx = 0
    for g in datos_grupos:
        r = rangos[idx: idx + len(g)]
        suma_rangos.append(r.sum())
        prom_rangos.append(r.mean())
        idx += len(g)

    H_sin = (12.0 / (N * (N + 1))) * sum(
        (Ri ** 2) / len(g) for Ri, g in zip(suma_rangos, datos_grupos)
    ) - 3 * (N + 1)

    _, counts = np.unique(todos, return_counts=True)
    tie_term = sum((t ** 3 - t) for t in counts if t > 1)
    C = 1 - tie_term / (N ** 3 - N) if (N ** 3 - N) > 0 else 1.0
    H_corr = H_sin / C if C > 0 else H_sin

    H_scipy, p_valor = stats.kruskal(*datos_grupos)

    return {
        "N": N,
        "suma_rangos": suma_rangos,
        "prom_rangos": prom_rangos,
        "H_sin": H_sin,
        "C": C,
        "H_corr": H_corr,
        "H_scipy": H_scipy,
        "p_valor": p_valor,
        "tie_term": tie_term,
    }


def chequear_forma_similar(datos_grupos):
    """Avisa si la razón IQR_max / IQR_min sugiere formas muy distintas."""
    iqrs = [float(stats.iqr(g)) for g in datos_grupos]
    positivos = [x for x in iqrs if x > 0]
    if not positivos:
        return iqrs, np.inf
    min_iqr = min(positivos)
    max_iqr = max(iqrs)
    ratio = max_iqr / min_iqr if min_iqr > 0 else np.inf
    return iqrs, ratio


# ------------------------------------------------------------
# 3. Ejecución de la prueba
# ------------------------------------------------------------
ejecutar = st.button("🔬 Ejecutar prueba de Kruskal-Wallis", type="primary")

if ejecutar:
    if not entradas_validas:
        st.error("Debes completar correctamente los datos de **todos** los grupos.")
        st.session_state.pop("resultado", None)
    else:
        res = calcular_kw(datos_grupos)
        iqrs, ratio_iqr = chequear_forma_similar(datos_grupos)
        gl = k - 1
        chi_critico = stats.chi2.ppf(1 - alpha, df=gl)
        # Decisión final por p-valor (coherencia, #7)
        rechaza_H0 = res["p_valor"] < alpha

        # Persistir resultados en session_state (#3)
        st.session_state["resultado"] = {
            "grupos": grupos,
            "nombres": nombres_grupos,
            "datos": [g.tolist() for g in datos_grupos],
            "k": k,
            "alpha": alpha,
            "gl": gl,
            "chi_critico": chi_critico,
            "rechaza_H0": rechaza_H0,
            "iqrs": iqrs,
            "ratio_iqr": ratio_iqr,
            **res,
        }

# ------------------------------------------------------------
# 4. Reporte (lee desde session_state para que persista)
# ------------------------------------------------------------
if "resultado" in st.session_state:
    R = st.session_state["resultado"]
    nombres_grupos = R["nombres"]
    datos_grupos = [np.asarray(g) for g in R["datos"]]
    alpha_r = R["alpha"]
    gl = R["gl"]
    chi_critico = R["chi_critico"]
    N = R["N"]
    H_sin = R["H_sin"]
    C = R["C"]
    H_corr = R["H_corr"]
    p_valor = R["p_valor"]

    st.header("3️⃣ Reporte completo")

    # Hipótesis
    st.subheader("📌 Planteamiento de hipótesis")
    st.markdown(
        "- **H₀:** Las distribuciones de los *k* grupos son iguales "
        "(bajo forma similar: medianas iguales).\n"
        "- **H₁:** Al menos una distribución difiere de las demás."
    )

    # Supuestos + chequeos (#5, #10)
    st.subheader("📋 Supuestos de la prueba y chequeos")
    st.markdown(
        "- Las muestras son **independientes** entre sí.\n"
        "- La variable es al menos **ordinal** (o continua).\n"
        "- Las distribuciones tienen **forma similar** (para interpretar en términos de medianas)."
    )
    tamaños = [len(g) for g in datos_grupos]
    if any(n < 5 for n in tamaños):
        st.warning(
            f"⚠️ Hay grupos con nᵢ < 5 (tamaños: {tamaños}). La aproximación χ² "
            "puede ser poco confiable; interpretar el p-valor con cautela."
        )
    if R["ratio_iqr"] == np.inf or R["ratio_iqr"] > 2:
        iqrs_fmt = [round(x, 3) for x in R["iqrs"]]
        st.warning(
            f"⚠️ Los rangos intercuartílicos difieren notablemente "
            f"(IQR = {iqrs_fmt}, razón max/min ≈ {R['ratio_iqr']:.2f}). "
            "El supuesto de **forma similar** puede no cumplirse: formular la "
            "conclusión sobre **distribuciones**, no sobre medianas."
        )

    # Resumen
    st.subheader("📊 Resumen de los grupos")
    resumen = pd.DataFrame({
        "n": tamaños,
        "Mediana": [np.median(g) for g in datos_grupos],
        "IQR": R["iqrs"],
        "Suma de rangos (Rᵢ)": R["suma_rangos"],
        "Rango promedio": R["prom_rangos"],
    }, index=nombres_grupos)
    st.dataframe(resumen.style.format({
        "Mediana": "{:.3f}",
        "IQR": "{:.3f}",
        "Suma de rangos (Rᵢ)": "{:.2f}",
        "Rango promedio": "{:.3f}",
    }))

    # Reconstrucción de rangos para las tablas de detalle
    todos_concat = np.concatenate(datos_grupos)
    rangos_concat = stats.rankdata(todos_concat)
    rangos_por_grupo = []
    _idx = 0
    for g in datos_grupos:
        rangos_por_grupo.append(rangos_concat[_idx: _idx + len(g)])
        _idx += len(g)

    # Tabla desplegable: observaciones ordenadas de menor a mayor con rangos
    with st.expander("🔽 Ver tabla de observaciones ordenadas con sus rangos"):
        etiquetas_grupo = []
        for nombre, g in zip(nombres_grupos, datos_grupos):
            etiquetas_grupo.extend([nombre] * len(g))
        df_orden = pd.DataFrame({
            "Valor": todos_concat,
            "Grupo": etiquetas_grupo,
            "Rango": rangos_concat,
        }).sort_values("Valor", kind="mergesort").reset_index(drop=True)
        df_orden.index = df_orden.index + 1
        df_orden.index.name = "Orden"
        st.dataframe(df_orden.style.format({
            "Valor": "{:.3f}",
            "Rango": "{:.2f}",
        }))

    # Tabla visible: datos y rangos por grupo con la suma Rᵢ al pie
    st.subheader("📑 Datos y rangos por grupo")
    max_n = max(len(g) for g in datos_grupos)
    columnas = {}
    for nombre, g, r in zip(nombres_grupos, datos_grupos, rangos_por_grupo):
        pad = max_n - len(g)
        columnas[f"{nombre} — Valor"] = list(g) + [np.nan] * pad
        columnas[f"{nombre} — Rango"] = list(r) + [np.nan] * pad
    df_datos_rangos = pd.DataFrame(columnas)
    df_datos_rangos.index = [str(i) for i in range(1, max_n + 1)]

    fila_suma = {}
    for nombre, r in zip(nombres_grupos, rangos_por_grupo):
        fila_suma[f"{nombre} — Valor"] = np.nan
        fila_suma[f"{nombre} — Rango"] = float(r.sum())
    df_suma = pd.DataFrame(fila_suma, index=["Σ Rᵢ"])

    df_datos_full = pd.concat([df_datos_rangos, df_suma])
    df_datos_full.index.name = "i"
    st.dataframe(df_datos_full.style.format(precision=3, na_rep=""))

    # Cálculo del estadístico — H sin y con corrección (#1)
    st.subheader("🧮 Cálculo del estadístico")
    st.markdown("**Fórmula sin corrección por empates:**")
    st.latex(r"H_{sin} = \frac{12}{N(N+1)} \sum_{i=1}^{k} \frac{R_i^2}{n_i} - 3(N+1)")
    st.markdown("**Factor de corrección por empates:**")
    st.latex(r"C = 1 - \frac{\sum_j (t_j^3 - t_j)}{N^3 - N} \qquad\Rightarrow\qquad H = \frac{H_{sin}}{C}")

    st.markdown(f"- **N total** = {N}")
    st.markdown(f"- **Grados de libertad (gl)** = k − 1 = {gl}")
    st.markdown(f"- **H sin corrección** = `{H_sin:.4f}`")
    st.markdown(
        f"- **Factor C** = `{C:.4f}` "
        + ("(hay empates → corrección aplicada)" if C < 1 else "(sin empates)")
    )
    st.markdown(f"- **H corregido** = `{H_corr:.4f}`")
    st.markdown(f"- **Valor crítico χ²** (α = {alpha_r}, gl = {gl}) = `{chi_critico:.4f}`")
    st.markdown(f"- **p-valor (scipy)** = `{p_valor:.4f}`")

    # Región de rechazo — eje x acotado (#11) y cierre de figura (#8)
    st.subheader("📈 Región de rechazo")
    x_max = max(chi_critico * 1.5, stats.chi2.ppf(0.999, df=gl))
    H_plot = min(H_corr, x_max)
    x = np.linspace(0, x_max, 500)
    y = stats.chi2.pdf(x, df=gl)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, label=f"χ² con gl = {gl}", color="steelblue")
    ax.fill_between(x, y, where=(x >= chi_critico),
                    color="salmon", alpha=0.5, label="Región de rechazo")
    etiqueta_H = f"H = {H_corr:.3f}"
    if H_corr > x_max:
        etiqueta_H += " (fuera de rango)"
    ax.axvline(H_plot, color="green", linestyle="--", label=etiqueta_H)
    ax.axvline(chi_critico, color="red", linestyle=":",
               label=f"χ² crítico = {chi_critico:.3f}")
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    # Decisión por p-valor (#7)
    st.subheader("✅ Decisión y conclusión")
    coinciden = (H_corr > chi_critico) == (p_valor < alpha_r)
    if not coinciden:
        st.warning(
            "⚠️ La aproximación χ² y el p-valor exacto no coinciden; "
            "se usa el **p-valor** como criterio final."
        )

    if R["rechaza_H0"]:
        st.success(
            f"**Se RECHAZA H₀** (p-valor = {p_valor:.4f} < α = {alpha_r}; "
            f"H = {H_corr:.4f} vs χ² crítico = {chi_critico:.4f})."
        )
        st.markdown(
            f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
            "existe evidencia estadística suficiente para afirmar que **al menos una** "
            "de las distribuciones difiere de las demás."
        )
    else:
        st.info(
            f"**NO se rechaza H₀** (p-valor = {p_valor:.4f} ≥ α = {alpha_r}; "
            f"H = {H_corr:.4f} vs χ² crítico = {chi_critico:.4f})."
        )
        st.markdown(
            f"**Conclusión:** Con un nivel de significancia del {int(alpha_r*100)}%, "
            "**no** existe evidencia estadística suficiente para afirmar que las "
            "distribuciones de los grupos sean diferentes."
        )
