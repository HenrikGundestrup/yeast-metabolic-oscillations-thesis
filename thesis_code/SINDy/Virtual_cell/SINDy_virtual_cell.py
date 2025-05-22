import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score
import pysindy as ps

# --- Define the 2-state glycolysis model ---
def glycolysis_model(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2])**p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2])**p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# --- Parameters ---
v, k1, K, n, k2 = 15, 1, 2, 3, 4
params = (v, k1, K, n, k2)
times = np.arange(0, 15, 0.01)
y0 = [0, 0]
dt = 0.01

# --- Simulate clean system ---
y_clean = odeint(glycolysis_model, t=times, y0=y0, args=(params,), rtol=1e-8)

# --- Add noise ---
noise_strength = 0.05
y_noisy = y_clean + noise_strength * np.random.randn(*y_clean.shape)

# --- Plot clean and noisy ---
plt.figure()
plt.plot(times, y_clean[:, 0], 'r', label='Clean $F6P$')
plt.plot(times, y_clean[:, 1], 'b', label='Clean $F16BP$')
plt.plot(times, y_noisy[:, 0], 'r--', label='Noisy $F6P$')
plt.plot(times, y_noisy[:, 1], 'b--', label='Noisy $F16BP$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.title("Clean vs Noisy Data")
plt.show()

# === SINDy Model on Clean Data ===
model_clean = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.05),
    feature_library=ps.PolynomialLibrary(degree=4),
)
model_clean.fit(y_clean, t=dt)
print("### Clean Data Model:")
model_clean.print()

x_sim_clean = model_clean.simulate(y0, times)

# Plot clean model results
plt.figure()
plt.plot(times, y_clean[:, 0], 'r', label='True $F6P$')
plt.plot(times, y_clean[:, 1], 'b', label='True $F16BP$')
plt.plot(times, x_sim_clean[:, 0], 'y--', label='SINDy $F6P$')
plt.plot(times, x_sim_clean[:, 1], 'g--', label='SINDy $F16BP$')
plt.legend()
plt.grid()
plt.title("SINDy on Clean Data")
plt.show()

# --- Phase portrait ---
plt.plot(y_clean[:, 0], y_clean[:, 1], 'r', label='True')
plt.plot(x_sim_clean[:, 0], x_sim_clean[:, 1], 'k--', label='SINDy')
plt.xlabel('$F6P$')
plt.ylabel('$F16BP$')
plt.legend()
plt.grid()
plt.title("Phase Portrait (Clean Data)")
plt.show()

# R2 Score for clean
r2_clean = (r2_score(y_clean[:, 0], x_sim_clean[:, 0]) + r2_score(y_clean[:, 1], x_sim_clean[:, 1])) / 2
print("Mean R2 (Clean Data):", r2_clean)

# === SINDy Model on Noisy Data ===
model_noisy = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.4),
    feature_library=ps.PolynomialLibrary(degree=4),
)
model_noisy.fit(y_noisy, t=dt)
print("### Noisy Data Model:")
model_noisy.print()

x_sim_noisy = model_noisy.simulate(y0, times)

# Plot noisy model results
plt.figure()
plt.plot(times, y_noisy[:, 0], 'r', label='Noisy $F6P$')
plt.plot(times, y_noisy[:, 1], 'b', label='Noisy $F16BP$')
plt.plot(times, x_sim_noisy[:, 0], 'y--', label='SINDy $F6P$')
plt.plot(times, x_sim_noisy[:, 1], 'g--', label='SINDy $F16BP$')
plt.legend()
plt.grid()
plt.title("SINDy on Noisy Data")
plt.show()

# --- Phase portrait ---
plt.plot(y_noisy[:, 0], y_noisy[:, 1], 'r', label='Noisy')
plt.plot(x_sim_noisy[:, 0], x_sim_noisy[:, 1], 'k--', label='SINDy')
plt.xlabel('$F6P$')
plt.ylabel('$F16BP$')
plt.legend()
plt.grid()
plt.title("Phase Portrait (Noisy Data)")
plt.show()

# R2 Score for noisy
r2_noisy = (r2_score(y_clean[:, 0], x_sim_noisy[:, 0]) + r2_score(y_clean[:, 1], x_sim_noisy[:, 1])) / 2
print("Mean R2 (Noisy Data):", r2_noisy)
