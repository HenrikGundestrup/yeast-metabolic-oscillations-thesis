import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydmd import DMD
from sklearn.metrics import r2_score

# Define the simplest 2-state glycolysis model
def Cell(y, t, p):
    dn = p[0] - p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# Time vector
times = np.arange(0, 15, 0.01)

# True model parameters
v, k1, K, n, k2 = 14, 1, 2, 3, 4

# Generate true data by simulating the system
y = odeint(Cell, y0=[0, 0], t=times, args=((v, k1, K, n, k2),), rtol=1e-8)

# Add noise to simulate observational data
noise_level = 0.05
yobs = y + np.random.normal(0, noise_level, y.shape)

# Function to construct delay embedding with future time steps
def future_delay_embedding(data, delay_steps):
    embedded_data = []
    for i in range(len(data) - delay_steps):
        stacked_row = data[i:i + delay_steps].flatten()
        embedded_data.append(stacked_row)
    return np.array(embedded_data)

# Apply future delay embedding
delay_steps = 150
yobs_embedded = future_delay_embedding(yobs, delay_steps)
times_embedded = times[:len(times) - delay_steps]

# Apply DMD to the future delay-embedded data
dmd = DMD(svd_rank=30)
dmd.fit(yobs_embedded.T)

# Reconstruct the data using DMD
y_dmd_embedded = dmd.reconstructed_data.real.T

# Plot DMD predictions vs. noisy observations
plt.plot(times_embedded, yobs[:len(times_embedded), 0], 'r', label='Observed $F6P(t)$', linewidth=3.0)
plt.plot(times_embedded, yobs[:len(times_embedded), 1], 'b', label='Observed $F16BP(t)$', linewidth=3.0)
plt.plot(times_embedded, y_dmd_embedded[:, 0], color='yellow', linestyle='--', label='DMD model $F6P(t)$')
plt.plot(times_embedded, y_dmd_embedded[:, 1], color='lightgreen', linestyle='--', label='DMD model $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.show()

# Phase plot comparison
plt.plot(y[:, 0], y[:, 1], 'r', label='True Trajectory')
plt.plot(y_dmd_embedded[:, 0], y_dmd_embedded[:, 1], 'k--', label='DMD Model Trajectory')
plt.plot(y[0, 0], y[0, 1], 'ro')
plt.plot(y_dmd_embedded[0, 0], y_dmd_embedded[0, 1], 'ko')
plt.xlabel('$F6P$')
plt.ylabel('$F16BP$')
plt.title('Phase Plot: True Trajectory vs DMD with Future Delay Embedding')
plt.legend()
plt.grid()
plt.show()

# Compute R² score
r2_f6p = r2_score(y_dmd_embedded[:, 0], yobs[:len(y_dmd_embedded), 0])
r2_f16bp = r2_score(y_dmd_embedded[:, 1], yobs[:len(y_dmd_embedded), 1])
r2_avg = (r2_f6p + r2_f16bp) / 2
print(f"R² F6P: {r2_f6p:.4f}, R² F16BP: {r2_f16bp:.4f}, Average R²: {r2_avg:.4f}")
