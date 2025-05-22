# Damped Harmonic Oscillator with SINDy (Clean and Noisy Data)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps

# --- System Parameters ---
gamma = 0.1  # Damping coefficient
omega = 1.0  # Natural frequency

# --- Define the damped harmonic oscillator ODE system ---
def damped_harmonic_oscillator(t, y):
    """
    ODE system for a damped harmonic oscillator:
        dx/dt = v
        dv/dt = -2γv - ω²x
    """
    x, v = y
    dxdt = v
    dvdt = -2 * gamma * v - omega**2 * x
    return [dxdt, dvdt]

# --- Simulation Setup ---
t_span = (0, 20)           # Time interval
dt = 0.01                  # Time step
t_eval = np.arange(*t_span, dt)  # Time points
y0 = [1, 0]                # Initial conditions: position=1, velocity=0

# --- Solve the ODE ---
sol = solve_ivp(damped_harmonic_oscillator, t_span, y0, t_eval=t_eval)

# Extract the clean solution
x_data = sol.y[0].reshape(-1, 1)  # Position
v_data = sol.y[1].reshape(-1, 1)  # Velocity
data = np.hstack((x_data, v_data))

# --- Add noise to the data ---
noise_strength = 0.05
x_noise = x_data + noise_strength * np.random.randn(*x_data.shape)
v_noise = v_data + noise_strength * np.random.randn(*v_data.shape)
data_noise = np.hstack((x_noise, v_noise))

# ===============================================================
# Part 1: Fit SINDy to clean data
# ===============================================================
print("### SINDy on CLEAN data ###")

# Instantiate and fit the SINDy model
model_clean = ps.SINDy()
model_clean.fit(data, t=dt)

# Print the discovered model
model_clean.print()

# Simulate using the identified model
sim_clean = model_clean.simulate(y0, t_eval)

# Plot true vs SINDy simulation (clean data)
plt.figure(figsize=(10, 5))
plt.plot(t_eval, data[:, 0], 'r', label='True Position', linewidth=2.5)
plt.plot(t_eval, data[:, 1], 'b', label='True Velocity', linewidth=2.5)
plt.plot(t_eval, sim_clean[:, 0], 'y--', label='SINDy Predicted Position')
plt.plot(t_eval, sim_clean[:, 1], 'lightgreen', linestyle='--', label='SINDy Predicted Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('SINDy on Clean Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ===============================================================
# Part 2: Fit SINDy to NOISY data
# ===============================================================
print("### SINDy on NOISY data ###")

# Instantiate SINDy model with sparse regression (STLSQ) and thresholding
model_noisy = ps.SINDy(optimizer=ps.STLSQ(threshold=0.15))
# You can optionally add: feature_library=ps.PolynomialLibrary(degree=2)

# Fit to noisy data
model_noisy.fit(data_noise, t=dt)

# Print discovered model
model_noisy.print()

# Simulate using the model fitted on noisy data
sim_noisy = model_noisy.simulate(y0, t_eval)

# Plot true (noisy) vs SINDy simulation (noisy data)
plt.figure(figsize=(10, 5))
plt.plot(t_eval, data_noise[:, 0], 'r', label='Noisy Position')
plt.plot(t_eval, data_noise[:, 1], 'b', label='Noisy Velocity')
plt.plot(t_eval, sim_noisy[:, 0], 'y--', label='SINDy Predicted Position')
plt.plot(t_eval, sim_noisy[:, 1], 'lightgreen', linestyle='--', label='SINDy Predicted Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('SINDy on Noisy Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
