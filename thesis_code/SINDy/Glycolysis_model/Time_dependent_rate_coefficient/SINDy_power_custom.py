import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score
import pysindy as ps
from pysindy.feature_library import PolynomialLibrary, CustomLibrary
from pysindy.optimizers import SR3

# -----------------------------
# Define system parameters
# -----------------------------
v, k1, K, p3, k2, b = 10, 1, 2, 3, 4, 0.1
p = (v, k1, K, p3, k2, b)

# -----------------------------
# Define the ODE system
# -----------------------------
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0] - p[4] * (t ** (-p[5])) * y[1]
    return [dn, dc]

# -----------------------------
# Simulate true system
# -----------------------------
times = np.arange(0.1, 15, 0.01)  # start from 0.1 to avoid division by 0
y0 = [0, 0]
y = odeint(Cell, y0, t=times, args=(p,), rtol=1e-8)

# -----------------------------
# Add log-normal noise (optional)
# -----------------------------
epsilon = 1e-6  # to avoid log(0)
yobs = np.random.lognormal(mean=np.log(y + epsilon), sigma=0)

# -----------------------------
# Plot observed data
# -----------------------------
plt.plot(times, yobs[:, 0], "r", label='Observed $F6P(t)$', linewidth=3.0)
plt.plot(times, yobs[:, 1], "b", label='Observed $F16BP(t)$', linewidth=3.0)
plt.plot(times, yobs[:, 0] + yobs[:, 1], 'g', label='Sum of $F6P + F16BP$', linewidth=3)
plt.ylabel('Concentration (mM)')
plt.xlabel('Time (s)')
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# Define custom feature library for SINDy
# -----------------------------
poly_library = PolynomialLibrary(degree=4)

feature_functions = [
    lambda x, t: (t + epsilon) ** (-b) * x[:, 1]
]

feature_names = [
    lambda x, t: f"t^{{-{b}}} * x1"
]

custom_library = CustomLibrary(
    library_functions=feature_functions,
    function_names=feature_names
)

combined_library = poly_library + custom_library

# -----------------------------
# Fit SINDy model
# -----------------------------
model = ps.SINDy(
    feature_library=combined_library,
    optimizer=SR3(threshold=0.1, nu=1e-2)
)

model.fit(y, t=times)
model.print()

# -----------------------------
# Simulate learned model
# -----------------------------
x_sim = model.simulate(y0, times)

# -----------------------------
# Plot model vs true
# -----------------------------
plt.plot(times, y[:, 0], "r", label='True $F6P(t)$', linewidth=3)
plt.plot(times, y[:, 1], "b", label='True $F16BP(t)$', linewidth=3)
plt.plot(times, x_sim[:, 0], "k-", label='SINDy $F6P(t)$')
plt.plot(times, x_sim[:, 1], "k--", label='SINDy $F16BP(t)$')
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")
plt.grid()
plt.show()

# -----------------------------
# Compute R² scores
# -----------------------------
r2_x0 = r2_score(y[:, 0], x_sim[:, 0])
r2_x1 = r2_score(y[:, 1], x_sim[:, 1])

print(f"R² score for F6P (x0): {r2_x0:.4f}")
print(f"R² score for F16BP (x1): {r2_x1:.4f}")
