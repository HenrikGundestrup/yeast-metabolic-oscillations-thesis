import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score
import pysindy as ps

# --- Glycolysis model definition ---
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2])**p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2])**p[3]) * y[0] - p[4] * (t**(-p[5])) * y[1]
    return [dn, dc]

# --- Ground truth parameters ---
v, k1, K, n, k2, b = 10, 1, 2, 3, 4, 0.3
params_true = (v, k1, K, n, k2)

# --- Time vector and simulation ---
times = np.arange(0.01, 15, 0.01)
y = odeint(Cell, t=times, y0=[0, 0], args=((v, k1, K, n, k2, b),), rtol=1e-8)

# --- Optionally add noise ---
# yobs = np.random.lognormal(mean=np.log(y), sigma=0.015)
yobs = y  # no noise

# --- Plot true data ---
plt.plot(times, yobs[:, 0], "r", label='Observed $F6P(t)$', linewidth=3.0)
plt.plot(times, yobs[:, 1], "b", label='Observed $F16BP(t)$', linewidth=3.0)
plt.ylabel('Concentration (mM)')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.show()

# --- SINDy setup ---
dt = 0.01
poly_order = 4
threshold = 0.05

model = ps.SINDy(
    optimizer=ps.SR3(threshold=0.1, nu=1e-2),
    feature_library=ps.PolynomialLibrary(degree=poly_order),
)
model.fit(yobs, t=dt)
model.print()

# --- Simulate SINDy model ---
x_sim = model.simulate([0, 0], times)

# --- Plot comparison ---
plt.plot(times, y[:, 0], "r", label='True $F6P(t)$', linewidth=4.0)
plt.plot(times, y[:, 1], "b", label='True $F16BP(t)$', linewidth=4.0)
plt.plot(times, x_sim[:, 0], '--', color='orange', label='SINDy $F6P(t)$')
plt.plot(times, x_sim[:, 1], '--', color='green', label='SINDy $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.show()

# --- R^2 scores ---
r2_f6p = r2_score(y[:, 0], x_sim[:, 0])
r2_f16bp = r2_score(y[:, 1], x_sim[:, 1])
print(f"R² score for F6P (x0): {r2_f6p:.4f}")
print(f"R² score for F16BP (x1): {r2_f16bp:.4f}")

# --- Parameter error estimate (optional) ---
# Replace this with true model identification if estimating parameters from coefficients
params_identified = (15.155, 1.030, 2.011, 3, 3.951)  # Placeholder values

relative_error = np.mean([abs((params_true[i] - params_identified[i]) / params_true[i]) for i in range(5)])
print(f"Mean relative parameter error: {100 * relative_error:.2f}%")
