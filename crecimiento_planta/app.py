import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

st.set_page_config(page_title="Crecimiento de una Planta", layout="centered")
st.title("🌱 Crecimiento de una Planta - Métodos Numéricos")

# Datos
x = np.array([0, 15, 30, 45, 60, 75, 90])
y = np.array([0, 4.16, 8.32, 12.01, 16.06, 17.77, 18.58])
df = pd.DataFrame({"Día": x, "Altura (cm)": y})

st.subheader("📊 Datos de crecimiento")
st.dataframe(df, use_container_width=True)

x_interp = st.slider("Elegí un día para estimar la altura:", min_value=0, max_value=90, step=1, value=38)

# Lagrange
def lagrange_interp(xi, x, y):
    n = len(x)
    result = 0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if j != i:
                term *= (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

# Spline cúbico con scipy
spline_cubico = CubicSpline(x, y)

# Regresión exponencial
def exp_reg(x, y):
    ln_y = np.log(np.where(y == 0, 0.001, y))
    A = np.vstack([x, np.ones(len(x))]).T
    b, ln_a = np.linalg.lstsq(A, ln_y, rcond=None)[0]
    a = np.exp(ln_a)
    return a, b

a, b = exp_reg(x, y)

# Método de Newton para sistema no lineal

def newton_non_linear():
    u, v = 1.0, 2.0
    for _ in range(6):
        f1 = v - u**3
        f2 = u**2 + v**2 - 1
        J = np.array([[-3*u**2, 1], [2*u, 2*v]])
        F = np.array([f1, f2])
        delta = np.linalg.solve(J, -F)
        u += delta[0]
        v += delta[1]
    return u, v

# Resultados
st.subheader("📈 Resultados de estimación")
u, v = newton_non_linear()
st.metric("Newton no lineal - u", f"{u:.5f}")
st.metric("Newton no lineal - v", f"{v:.5f}")
col1, col2 = st.columns(2)

with col1:
    lag_val = lagrange_interp(x_interp, x, y)
    st.metric("Lagrange", f"{lag_val:.3f} cm")

with col2:
    spline_val = spline_cubico(x_interp)
    st.metric("Spline cúbico", f"{spline_val:.3f} cm")

st.metric("Regresión exponencial", f"{a * np.exp(b * x_interp):.3f} cm")



# Gráfica
st.subheader("📉 Gráfica comparativa")
x_dense = np.linspace(0, 90, 300)
lagrange_y = [lagrange_interp(xi, x, y) for xi in x_dense]
spline_y = spline_cubico(x_dense)
exp_y = a * np.exp(b * x_dense)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, y, 'o', label="Datos reales")
ax.plot(x_dense, lagrange_y, '--', label="Lagrange")
ax.plot(x_dense, spline_y, ':', label="Spline cúbico")
ax.plot(x_dense, exp_y, '-.', label="Regresión exp")
ax.set_xlabel("Día")
ax.set_ylabel("Altura (cm)")
ax.set_title("Comparación de métodos")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption("Desarrollado por Carlos. Métodos implementados: Lagrange, Spline cúbico, Regresión Exponencial, Newton no lineal.")
