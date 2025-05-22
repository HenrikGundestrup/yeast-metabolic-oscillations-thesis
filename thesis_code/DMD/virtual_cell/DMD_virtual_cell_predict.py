import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydmd import DMD
from sklearn.metrics import r2_score

# Define the 2-state glycolysis model
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# Time vector
times = np.arange(0, 15, 0.01)

# Parameters
v, k1, K, n, k2 = 14, 1, 2, 3, 4

# Simulate system
y = odeint(Cell, y0=[0, 0], t=times, args=((v, k1, K, n, k2),), rtol=1e-8)

# Add noise
noise_level = 0.0
yobs = y + np.random.normal(0, noise_level, y.shape)

# Future delay embedding
def future_delay_embedding(data, delay_steps):
    return np.array([data[i:i + delay_steps].flatten() for i in range(len(data) - delay_steps)])

delay_steps = 150
yobs = yobs[:len(yobs)//2]
yobs_embedded = future_delay_embedding(yobs, delay_steps)
times_embedded = times[:len(times) - delay_steps]

# Split into training
midpoint = len(yobs_embedded) // 2
yobs_embedded_train = yobs_embedded[:midpoint]
times_embedded_train = times_embedded[:midpoint]

# Fit DMD
dmd = DMD(svd_rank=30)
dmd.fit(yobs_embedded.T)
dmd.dmd_time['tend'] = 1349
y_dmd_embedded_full = dmd.reconstructed_data.real.T
reconstructed_times = times_embedded[:y_dmd_embedded_full.shape[0]]

# Plot time-series
plt.plot(times_embedded, y[:len(times_embedded), 0], 'r', label='True $F6P(t)$', linewidth=3)
plt.plot(times_embedded, y[:len(times_embedded), 1], 'b', label='True $F16BP(t)$', linewidth=3)
plt.plot(reconstructed_times, y_dmd_embedded_full[:, 0], '--', color='yellow', label='DMD $F6P(t)$')
plt.plot(reconstructed_times, y_dmd_embedded_full[:, 1], '--', color='lightgreen', label='DMD $F16BP(t)$')
plt.axvline(0.01 * len(y)//2, color='g', linestyle='--', label='Train/Test Split')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.show()

# Phase plot
plt.plot(y[:, 0], y[:, 1], 'r', label='True Trajectory')
plt.plot(y_dmd_embedded_full[:, 0], y_dmd_embedded_full[:, 1], 'k--', label='DMD Trajectory')
plt.xlabel('$F6P$')
plt.ylabel('$F16BP$')
plt.title('Phase Plot: DMD Extrapolation')
plt.legend()
plt.grid()
plt.show()

# R2 score
r2_f6p = r2_score(y_dmd_embedded_full[:, 0], y[:len(y_dmd_embedded_full), 0])
r2_f16bp = r2_score(y_dmd_embedded_full[:, 1], y[:len(y_dmd_embedded_full), 1])
print((r2_f6p + r2_f16bp) / 2)
