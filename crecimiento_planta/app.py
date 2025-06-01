import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Crecimiento de una Planta", layout="centered")
st.title("🌱 Crecimiento de una Planta - Métodos Numéricos")

# Datos
x = np.array([0, 15, 30, 45, 60, 75, 90])
y = np.array([0, 4.16, 8.32, 12.01, 16.06, 17.77, 18.58])
df = pd.DataFrame({"Día": x, "Altura (cm)": y})

st.subheader("📊 Datos de crecimiento")
st.dataframe(df, use_container_width=True)

x_interp = st.slider("Elegí un día para estimar la altura:", min_value=0, max_value=90, step=1, value=38)

# Newton (diferencias divididas)
def newton_divided_differences(x_points, y_points):
    n = len(x_points)
    f = np.zeros((n, n))
    f[:, 0] = y_points
    for j in range(1, n):
        for i in range(n - j):
            f[i, j] = (f[i+1, j-1] - f[i, j-1]) / (x_points[i+j] - x_points[i])
    return f[0, :]

def newton_polynomial_evaluate(coefficients, x_points, x_eval):
    n = len(coefficients)
    result = coefficients[0]
    product_term = 1.0
    for i in range(1, n):
        product_term *= (x_eval - x_points[i-1])
        result += coefficients[i] * product_term
    return result

newton_coeffs = newton_divided_differences(x, y)
newton_val = newton_polynomial_evaluate(newton_coeffs, x, x_interp)

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

a_exp, b_exp = exp_reg(x, y)
exp_val = a_exp * np.exp(b_exp * x_interp)

# Regresión lineal
x_reshape = x.reshape(-1, 1)
model_linear = LinearRegression()
model_linear.fit(x_reshape, y)
slope = model_linear.coef_[0]
intercept = model_linear.intercept_
r2_linear = r2_score(y, model_linear.predict(x_reshape))
linear_val = model_linear.predict(np.array([[x_interp]]))[0]

# Resultados
st.subheader("📈 Resultados de estimación")

st.metric("Interpolación de Newton", f"{newton_val:.3f} cm")
col1, col2 = st.columns(2)

with col1:
    lag_val = lagrange_interp(x_interp, x, y)
    st.metric("Lagrange", f"{lag_val:.3f} cm")

with col2:
    spline_val = spline_cubico(x_interp)
    st.metric("Spline cúbico", f"{spline_val:.3f} cm")

st.metric("Regresión lineal", f"{linear_val:.3f} cm\nR² = {r2_linear:.3f}")

# Gráfica
st.subheader("📉 Gráfica comparativa")
x_dense = np.linspace(0, 90, 300)
lagrange_y = [lagrange_interp(xi, x, y) for xi in x_dense]
spline_y = spline_cubico(x_dense)
exp_y = a_exp * np.exp(b_exp * x_dense)
newton_y = [newton_polynomial_evaluate(newton_coeffs, x, xi) for xi in x_dense]
linear_y = model_linear.predict(x_dense.reshape(-1, 1))

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x, y, 'o', label="Datos reales")
ax.plot(x_dense, newton_y, label="Newton")
ax.plot(x_dense, lagrange_y, '--', label="Lagrange")
ax.plot(x_dense, spline_y, ':', label="Spline cúbico")
ax.plot(x_dense, exp_y, '-.', label="Regresión exponencial")
ax.plot(x_dense, linear_y, '-', label="Regresión lineal")
ax.set_xlabel("Día")
ax.set_ylabel("Altura (cm)")
ax.set_title("Comparación de métodos")
ax.legend()
ax.grid(True)

st.pyplot(fig)

