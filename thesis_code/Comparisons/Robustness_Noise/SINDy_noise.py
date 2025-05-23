# SINDy Robustness Test on Noisy 2-State Glycolysis System

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from sklearn.metrics import r2_score

# ----------------------------------------
# 1. Define ODE System (Simplified Glycolysis Model)
# ----------------------------------------
def glycolysis_ode(y, t, p):
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = 1 + (F16BP / K) ** n
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# Simulation time and model parameters
t = np.arange(0, 15, 0.01)
params = (15, 1, 2, 3, 4)

# Simulate ODE to generate ground truth data
y_true = odeint(glycolysis_ode, y0=[0, 0], t=t, args=(params,), rtol=1e-8)

# ----------------------------------------
# 2. Function to Test Noise Levels and Fit SINDy
# ----------------------------------------
def test_noise_levels(noise_levels):
    r2_scores_f6p = []
    r2_scores_f16bp = []

    for noise_std in noise_levels:
        # Add Gaussian noise to synthetic data
        noise = np.random.normal(0, noise_std, y_true.shape)
        y_noisy = y_true + noise

        # Configure and fit SINDy model
        dt = 0.01
        poly_order = 4
        threshold = 0.05
        model = ps.SINDy(
            optimizer=ps.STLSQ(threshold=threshold),
            feature_library=ps.PolynomialLibrary(degree=poly_order),
        )
        model.fit(y_noisy, t=dt)

        # Simulate SINDy model from same initial condition
        y_sim = model.simulate([0, 0], t)

        # Compute R² score for both state variables
        r2_f6p = r2_score(y_noisy[:, 0], y_sim[:, 0])
        r2_f16bp = r2_score(y_noisy[:, 1], y_sim[:, 1])
        r2_scores_f6p.append(r2_f6p)
        r2_scores_f16bp.append(r2_f16bp)

        # Plot comparison for each noise level
        plt.plot(t, y_noisy[:, 0], "r", label='Noisy $F6P(t)$', linewidth=4)
        plt.plot(t, y_noisy[:, 1], "b", label='Noisy $F16BP(t)$', linewidth=4)
        plt.plot(t, y_sim[:, 0], "k-", label='SINDy $F6P(t)$')
        plt.plot(t, y_sim[:, 1], "k--", label='SINDy $F16BP(t)$')
        plt.xlabel('Time (s)')
        plt.ylabel('Concentration (mM)')
        plt.grid()
        plt.legend()
        plt.title(f'Noise std = {noise_std:.3f}')
        plt.show()

    return r2_scores_f6p, r2_scores_f16bp

# ----------------------------------------
# 3. Run SINDy on Varying Noise Levels
# ----------------------------------------
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2]
r2_f6p, r2_f16bp = test_noise_levels(noise_levels)

# Compute average R² across both variables
r2_avg = (np.array(r2_f6p) + np.array(r2_f16bp)) / 2

# ----------------------------------------
# 4. Plot R² vs Noise Level
# ----------------------------------------
plt.plot(noise_levels, r2_avg, marker='o', linestyle='-', color='blue', label='Average R²')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('R² Score')
plt.grid()
plt.legend()
plt.title('SINDy Model Robustness to Noise')
plt.show()

# ----------------------------------------
# 5. Print R² Scores for Each Noise Level
# ----------------------------------------
for noise_std, r2_0, r2_1 in zip(noise_levels, r2_f6p, r2_f16bp):
    print(f"Noise std: {noise_std:.3f}, R² ($F6P$): {r2_0:.4f}, R² ($F16BP$): {r2_1:.4f}")
