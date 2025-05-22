import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps

# --- Define the harmonic oscillator dynamics ---
def harmonic_oscillator(t, y):
    """
    Defines the ODEs for a simple harmonic oscillator:
        dx/dt = v
        dv/dt = -x
    
    Parameters:
        t : float
            Time (not explicitly used as the system is autonomous)
        y : array_like
            State vector [position, velocity]
    
    Returns:
        dydt : list
            Derivatives [velocity, acceleration]
    """
    return [y[1], -y[0]]

# --- Simulation parameters ---
t_span = (0, 10)  # Time interval for integration (in seconds)
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Evaluation points
y0 = [1, 0]  # Initial condition: position = 1, velocity = 0

# --- Solve the ODE system ---
solution = solve_ivp(harmonic_oscillator, t_span, y0, t_eval=t_eval)
t = solution.t
y = solution.y.T  # Transpose to shape (n_samples, n_features)

# --- Add noise to the clean data ---
noise_strength = 0.1  # Noise standard deviation
noisy_y = y + noise_strength * np.random.randn(*y.shape)  # Add Gaussian noise

# --- Plot the clean and noisy data ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t, y[:, 0], 'r', label='Position (clean)')
plt.plot(t, y[:, 1], 'b', label='Velocity (clean)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Clean Data')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t, noisy_y[:, 0], 'r', label='Position (noisy)')
plt.plot(t, noisy_y[:, 1], 'b', label='Velocity (noisy)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Noisy Data')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Apply SINDy to identify governing equations ---
# Instantiate the SINDy model with linear library and sparse regression
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=1)
)

# Fit the model using noisy data
model.fit(noisy_y, t=t)

# Print the discovered equations
print("Discovered SINDy model equations:")
model.print()

# --- Simulate the system using the learned SINDy model ---
simulated_y = model.simulate(y0, t)

# --- Plot the true and predicted trajectories ---
plt.figure(figsize=(10, 5))
plt.plot(t, y[:, 0], 'r', label='True Position', linewidth=3)
plt.plot(t, y[:, 1], 'b', label='True Velocity', linewidth=3)
plt.plot(t, simulated_y[:, 0], 'y--', label='SINDy Position')
plt.plot(t, simulated_y[:, 1], 'lightgreen', linestyle='--', label='SINDy Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('True vs SINDy Simulated Trajectories')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
