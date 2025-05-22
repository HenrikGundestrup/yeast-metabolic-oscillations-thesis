import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from scipy.integrate import solve_ivp

# Define constants
k = 1.0
m = 1.0
omega = np.sqrt(k / m)

# Define system
def simple_harmonic_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# Initial conditions and time span
x0, v0 = 1.0, 0.0
initial_conditions = [x0, v0]
t_span = (0, 10)
t_eval = np.linspace(0, 10, 1000)

# Solve system
solution = solve_ivp(simple_harmonic_oscillator, t_span, initial_conditions, t_eval=t_eval)
t = solution.t
y = solution.y.T

# Delay embedding
d = 2
embedded_data = np.zeros((y.shape[0] - d, d * y.shape[1]))
for i in range(d, y.shape[0]):
    embedded_data[i - d] = y[i - d:i].flatten()

# Apply DMD
dmd = DMD(svd_rank=2)
dmd.fit(embedded_data)
reconstructed_data = dmd.reconstructed_data.real
t_reconstructed = t[d:]

# Plot true dynamics
plt.plot(t, y[:, 0], 'r', label='True Position x', linewidth=3)
plt.plot(t, y[:, 1], 'b', label='True Velocity v', linewidth=3)

# Plot DMD reconstruction
plt.plot(t_reconstructed, reconstructed_data[:, 0], color='yellow', linestyle='--', label='DMD Position x')
plt.plot(t_reconstructed, reconstructed_data[:, 1], color='lightgreen', linestyle='--', label='DMD Velocity v')
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Print DMD results
print("DMD Eigenvalues:", dmd.eigs)
print("DMD Modes:\n", dmd.modes)
