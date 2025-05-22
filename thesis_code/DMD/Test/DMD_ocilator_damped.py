# Damped Harmonic Oscillator using DMD

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pydmd import DMD

# Parameters
gamma = 0.1  # Damping coefficient
omega = 1.0  # Natural frequency

# Damped harmonic oscillator system
def damped_harmonic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -2 * gamma * v - omega**2 * x
    return [dxdt, dvdt]

# Time domain
t_span = (0, 20)
dt = 0.01
t_eval = np.arange(t_span[0], t_span[1], dt)

# Initial conditions
y0 = [10, 1]

# Solve ODE
sol = solve_ivp(damped_harmonic_oscillator, t_span, y0, t_eval=t_eval)
x_data = sol.y[0].reshape(-1, 1)  # Position
v_data = sol.y[1].reshape(-1, 1)  # Velocity

# Prepare data for DMD
data = np.hstack((x_data, v_data)).T  # Shape: (features, time steps)

# Apply DMD
dmd = DMD(svd_rank=2)
dmd.fit(data)

# Reconstruct system
dmd_reconstructed = dmd.reconstructed_data.real.T  # Shape: (time steps, features)

# Plot original and DMD reconstructed data
plt.figure(figsize=(10, 4))
plt.plot(t_eval, x_data, 'r', label='True Position x', linewidth=2)
plt.plot(t_eval, v_data, 'b', label='True Velocity v', linewidth=2)
plt.plot(t_eval, dmd_reconstructed[:, 0], '--', color='orange', label='DMD Position x')
plt.plot(t_eval, dmd_reconstructed[:, 1], '--', color='green', label='DMD Velocity v')
plt.xlabel('Time (s)')
plt.ylabel('Position / Velocity')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print DMD output
print("DMD Eigenvalues:", dmd.eigs)
print("DMD Modes:\n", dmd.modes)

# Plot eigenvalues in complex plane
eigenvalues = dmd.eigs
plt.figure(figsize=(6, 6))
plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', s=100)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Eigenvalue Plot for Damped Harmonic Oscillator")
plt.grid(True)

# Unit circle
unit_circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', linewidth=1)
plt.gca().add_artist(unit_circle)

# Annotate eigenvalues
for i, val in enumerate(eigenvalues):
    plt.annotate(f"λ{i+1} = {val:.2f}", (val.real, val.imag))

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.tight_layout()
plt.show()
