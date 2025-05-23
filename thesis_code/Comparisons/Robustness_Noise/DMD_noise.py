# DMD Robustness Test on Noisy 2-State Glycolysis System

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydmd import DMD
from sklearn.metrics import r2_score

# ----------------------------------------
# 1. Define Simplified Glycolysis Model (2-State ODE)
# ----------------------------------------
def glycolysis_ode(y, t, p):
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = 1 + (F16BP / K) ** n
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# Simulation time vector and model parameters
t = np.arange(0, 15, 0.01)
params = (15, 1, 2, 3, 4)

# Simulate clean system dynamics
y_true = odeint(glycolysis_ode, y0=[0, 0], t=t, args=(params,), rtol=1e-8)

# ----------------------------------------
# 2. Helper Function: Future Delay Embedding
# ----------------------------------------
def future_delay_embedding(data, delay_steps):
    """
    Creates a delay-embedded version of the input time series by stacking future steps.
    Returns a 2D array of shape (n_samples - delay_steps, delay_steps * n_states)
    """
    return np.array([data[i:i + delay_steps].flatten() for i in range(len(data) - delay_steps)])

# ----------------------------------------
# 3. Define Noise Levels and DMD Evaluation Loop
# ----------------------------------------
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
delay_steps = 150  # Number of future time steps in the delay embedding

r2_scores_f6p = []
r2_scores_f16bp = []

for noise_std in noise_levels:
    # Add Gaussian noise to the clean data
    y_noisy = np.random.normal(loc=y_true, scale=noise_std)

    # Delay embedding on noisy data
    y_embedded = future_delay_embedding(y_noisy, delay_steps)
    t_embedded = t[:len(t) - delay_steps]

    # Fit DMD to the delay-embedded data
    dmd = DMD(svd_rank=30)
    dmd.fit(y_embedded.T)  # Transpose for shape (n_features, n_snapshots)

    # Reconstruct data from DMD
    y_dmd_recon = dmd.reconstructed_data.real.T  # Transpose back to (n_snapshots, n_features)

    # Extract original state dimensions
    y_dmd_f6p = y_dmd_recon[:, 0]
    y_dmd_f16bp = y_dmd_recon[:, 1]

    # Compute R² scores on truncated noisy data (same length as reconstructed output)
    r2_f6p = r2_score(y_noisy[:len(y_dmd_f6p), 0], y_dmd_f6p)
    r2_f16bp = r2_score(y_noisy[:len(y_dmd_f16bp), 1], y_dmd_f16bp)

    r2_scores_f6p.append(r2_f6p)
    r2_scores_f16bp.append(r2_f16bp)

# ----------------------------------------
# 4. Plot R² vs Noise Level
# ----------------------------------------
r2_avg = (np.array(r2_scores_f6p) + np.array(r2_scores_f16bp)) / 2

plt.plot(noise_levels, r2_avg, marker='o', linestyle='-', color='blue', label='Average R²')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('R² Score')
plt.grid(True)
plt.legend()
plt.title('DMD Model Robustness to Noise')
plt.show()

# ----------------------------------------
# 5. Print R² Scores for Each State Variable
# ----------------------------------------
for i, noise_std in enumerate(noise_levels):
    print(f"Noise std: {noise_std:.3f}, R² (F6P): {r2_scores_f6p[i]:.4f}, R² (F16BP): {r2_scores_f16bp[i]:.4f}")
