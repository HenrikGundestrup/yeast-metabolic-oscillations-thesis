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
t_span = (0, 100)
t_eval = np.linspace(0, 10, 1000)

# Solve system
solution = solve_ivp(simple_harmonic_oscillator, t_span, initial_conditions, t_eval=t_eval)
t = solution.t
y = solution.y.T

# Add noise
noise_strength = 0.1
noisy_y = y + noise_strength * np.random.randn(*y.shape)

# Delay embedding
d = 50
embedded_data = np.zeros((noisy_y.shape[0] - d, d * noisy_y.shape[1]))
for i in range(d, noisy_y.shape[0]):
    embedded_data[i - d] = noisy_y[i - d:i].flatten()

# Apply DMD
dmd = DMD(svd_rank=2)
dmd.fit(embedded_data)
reconstructed_data = dmd.reconstructed_data.real
t_reconstructed = t[d:]

# Plot noisy input data
plt.plot(t, noisy_y[:, 0], 'r', label='Noisy Position x')
plt.plot(t, noisy_y[:, 1], 'b', label='Noisy Velocity v')
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s)')
plt.legend()
plt.grid()

# Plot DMD reconstruction
plt.plot(t[:-d], reconstructed_data[:, 0], color='yellow', linestyle='--', label='DMD Position x')
plt.plot(t[:-d], reconstructed_data[:, 1], color='lightgreen', linestyle='--', label='DMD Velocity v')
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s)')
plt.legend()
plt.xlim((0, 10))
plt.tight_layout()
plt.show()

# Print eigenvalues and modes
print("DMD Eigenvalues:", dmd.eigs)
print("DMD Modes:\n", dmd.modes)

# Plot eigenvalues
eigenvalues = dmd.eigs
plt.figure(figsize=(6, 6))
plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', s=100)
plt.axhline(0, color='black', lw=0.5)
plt.axvline(0, color='black', lw=0.5)
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Eigenvalue Plot for DMD")
plt.grid(True)

# Unit circle
unit_circle = plt.Circle((0, 0), 1, color='red', fill=False, linestyle='--', linewidth=1)
plt.gca().add_artist(unit_circle)

# Annotate eigenvalues
for i, eigenvalue in enumerate(eigenvalues):
    plt.annotate(f"λ{i+1} = {eigenvalue:.2f}", (eigenvalue.real, eigenvalue.imag))

plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.show()
