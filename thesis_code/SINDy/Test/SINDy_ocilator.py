import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pysindy as ps
from sklearn.metrics import r2_score

# --- Define system parameters ---
k = 1.0  # Spring constant (N/m)
m = 1.0  # Mass of the oscillator (kg)
omega = np.sqrt(k / m)  # Angular frequency (rad/s)

# --- Define the system of ODEs for the simple harmonic oscillator ---
def simple_harmonic_oscillator(t, y):
    """
    Computes derivatives for the simple harmonic oscillator.
    
    Parameters:
        t : float
            Current time (not used explicitly since system is autonomous)
        y : array_like
            Current state vector [position, velocity]
    
    Returns:
        dydt : list
            Derivatives [velocity, acceleration]
    """
    x, v = y
    dxdt = v
    dvdt = -omega**2 * x
    return [dxdt, dvdt]

# --- Initial conditions ---
x0 = 1.0  # Initial position (m)
v0 = 0.0  # Initial velocity (m/s)
initial_conditions = [x0, v0]

# --- Time span and evaluation points ---
t_span = (0, 10)  # Start and end times (seconds)
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # Time points where solution is computed

# --- Solve the ODE system ---
solution = solve_ivp(simple_harmonic_oscillator, t_span, initial_conditions, t_eval=t_eval)

# --- Plot true solution ---
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], label='Position (x)')
plt.plot(solution.t, solution.y[1], label='Velocity (v)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Simple Harmonic Oscillator - True Solution')
plt.legend()
plt.grid(True)
plt.show()

# --- Prepare data for SINDy ---
t = solution.t
y = solution.y.T  # Transpose so shape is (n_samples, n_features)

# --- Create and configure SINDy model ---
model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.1),
    feature_library=ps.PolynomialLibrary(degree=1)  # Use linear features only
)

# --- Fit the SINDy model on the simulated data ---
model.fit(y, t=t)

# --- Print the discovered equations ---
print("Discovered SINDy model equations:")
model.print()

# --- Simulate the system using the SINDy model ---
x_sim = model.simulate(initial_conditions, t)

# --- Plot true vs predicted trajectories ---
plt.figure(figsize=(10, 5))
plt.plot(solution.t, solution.y[0], color='red', label='True Position (x)', linewidth=3)
plt.plot(solution.t, solution.y[1], color='blue', label='True Velocity (v)', linewidth=3)
plt.plot(t, x_sim[:, 0], color='orange', linestyle='--', label='SINDy Predicted Position', linewidth=2)
plt.plot(t, x_sim[:, 1], color='green', linestyle='--', label='SINDy Predicted Velocity', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Position (m), Velocity (m/s)')
plt.title('True vs SINDy Predicted Trajectories')
plt.legend()
plt.grid(True)
plt.show()

# --- Evaluate model performance using R^2 score ---
r2_x0 = r2_score(solution.y[0, :], x_sim[:, 0])
r2_x1 = r2_score(solution.y[1, :], x_sim[:, 1])
print(f'R^2 score for Position: {r2_x0:.4f}')
print(f'R^2 score for Velocity: {r2_x1:.4f}')
