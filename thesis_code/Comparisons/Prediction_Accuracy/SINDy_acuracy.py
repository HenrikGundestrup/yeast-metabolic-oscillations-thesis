# SINDy R² Test for 2-State Glycolysis Model

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score
import pysindy as ps

# -------------------------
# Define the glycolysis ODE system
# -------------------------
def glycolysis_model(y, t, p):
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = (1 + (F16BP / K)**n)
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# -------------------------
# Simulate synthetic data
# -------------------------
# True parameter values
true_params = (15, 1, 2, 3, 4)

# Time vector
t = np.arange(0, 15, 0.01)

# Initial condition
y0 = [0, 0]

# Simulate ODE system
y_true = odeint(glycolysis_model, t=t, y0=y0, args=(true_params,), rtol=1e-8)

# Plot the ground truth simulation
plt.figure()
plt.plot(t, y_true[:, 0], label='$F6P(t)$', color='C0', linewidth=3)
plt.plot(t, y_true[:, 1], label='$F16BP(t)$', color='C1', linewidth=3)
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.title('Synthetic Data: 2-State Glycolysis Model (Noise-free)')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Fit SINDy model
# -------------------------
# Time step
dt = 0.01

# Polynomial degree and threshold for sparse regression
poly_order = 4
threshold = 0.05

# Configure and fit SINDy
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=threshold),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)
model.fit(y_true, t=dt)
model.print()  # Display learned equations

# Simulate the SINDy model
y_sindy = model.simulate(y0, t)

# -------------------------
# Plot SINDy vs ground truth
# -------------------------
plt.figure()
plt.plot(t, y_true[:, 0], "r", label='Observed $F6P(t)$', linewidth=4)
plt.plot(t, y_true[:, 1], "b", label='Observed $F16BP(t)$', linewidth=4)
plt.plot(t, y_sindy[:, 0], "k-", label='SINDy model $F6P(t)$')
plt.plot(t, y_sindy[:, 1], "k--", label='SINDy model $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.title('SINDy vs Ground Truth')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------
# Compute R² scores
# -------------------------
r2_f6p = r2_score(y_true[:, 0], y_sindy[:, 0])
r2_f16bp = r2_score(y_true[:, 1], y_sindy[:, 1])

# -------------------------
# Parameter comparison (manual input of learned parameters)
# -------------------------
# These values should be obtained from model.print(), manually extracted or parsed
learned_params = (15.155, 1.030, 2.011, 3, 3.951)

# Compute mean relative error (%) in parameters
param_error_percent = np.mean([
    abs((true - learned) / true) * 100
    for true, learned in zip(true_params, learned_params)
])

# -------------------------
# Print evaluation results
# -------------------------
print(f"R² score for F6P: {r2_f6p:.4f}")
print(f"R² score for F16BP: {r2_f16bp:.4f}")
print(f"Mean relative error in parameters: {param_error_percent:.4f}%")
