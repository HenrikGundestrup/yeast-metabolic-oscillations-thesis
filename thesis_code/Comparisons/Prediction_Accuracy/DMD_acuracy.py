# DMD R² Test for 2-State Glycolysis Model with Future Delay Embedding

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydmd import DMD
from sklearn.metrics import r2_score

# -------------------------
# Define the glycolysis model ODE system
# -------------------------
def glycolysis_model(y, t, p):
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = (1 + (F16BP / K) ** n)
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# -------------------------
# Generate synthetic data
# -------------------------
# Time vector
t = np.arange(0, 15, 0.01)

# True parameter values
true_params = (14, 1, 2, 3, 4)

# Initial condition
y0 = [0, 0]

# Simulate ODE system
y_true = odeint(glycolysis_model, y0=y0, t=t, args=(true_params,), rtol=1e-8)

# Use clean data for DMD (can be replaced with noisy version if needed)
y_obs = y_true

# -------------------------
# Construct delay-embedded snapshot matrix using future values
# -------------------------
def future_delay_embedding(data, delay_steps):
    """
    Create future delay embedding matrix by stacking future time steps.
    Parameters:
        data (ndarray): Original data (n_samples x n_features).
        delay_steps (int): Number of future steps in each snapshot.
    Returns:
        embedded_data (ndarray): Delay-embedded data (n_snapshots x delay_steps * n_features).
    """
    embedded_data = [
        data[i:i + delay_steps].flatten()
        for i in range(len(data) - delay_steps)
    ]
    return np.array(embedded_data)

# Apply delay embedding
delay_steps = 150
y_embedded = future_delay_embedding(y_obs, delay_steps)

# Adjust time vector to match embedded data
t_embedded = t[:len(t) - delay_steps]

# -------------------------
# Apply DMD to embedded data
# -------------------------
dmd = DMD(svd_rank=30)
dmd.fit(y_embedded.T)  # DMD expects input as (n_features, n_snapshots)

# Reconstruct dynamics from DMD
y_dmd_reconstructed = dmd.reconstructed_data.real.T

# -------------------------
# Plot DMD vs true data
# -------------------------
plt.figure()
plt.plot(t_embedded, y_obs[:len(t_embedded), 0], 'r', label='Observed $F6P(t)$', linewidth=4)
plt.plot(t_embedded, y_obs[:len(t_embedded), 1], 'b', label='Observed $F16BP(t)$', linewidth=4)
plt.plot(t_embedded, y_dmd_reconstructed[:, 0], 'k-', label='DMD model $F6P(t)$')
plt.plot(t_embedded, y_dmd_reconstructed[:, 1], 'k--', label='DMD model $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.title('DMD with Future Delay Embedding')
plt.show()

# -------------------------
# Phase plot comparison
# -------------------------
plt.figure()
plt.plot(y_true[:, 0], y_true[:, 1], 'r', label='True Trajectory')
plt.plot(y_dmd_reconstructed[:, 0], y_dmd_reconstructed[:, 1], 'k--', label='DMD Model Trajectory')
plt.xlabel('$F6P$')
plt.ylabel('$F16BP$')
plt.legend()
plt.grid()
plt.title('Phase Plot: True vs DMD Trajectory')
plt.show()

# -------------------------
# R² score evaluation
# -------------------------
r2_f6p = r2_score(y_obs[:len(y_dmd_reconstructed), 0], y_dmd_reconstructed[:, 0])
r2_f16bp = r2_score(y_obs[:len(y_dmd_reconstructed), 1], y_dmd_reconstructed[:, 1])

# -------------------------
# Print results
# -------------------------
print(f"R² score for F6P: {r2_f6p:.4f}")
print(f"R² score for F16BP: {r2_f16bp:.4f}")
