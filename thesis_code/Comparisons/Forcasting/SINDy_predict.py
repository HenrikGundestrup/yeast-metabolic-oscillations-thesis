import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pysindy as ps
from sklearn.metrics import r2_score

# -------------------------------------------
# 1. Define the 2-state glycolysis ODE system
# -------------------------------------------
def glycolysis_ode(y, t, p):
    """
    Simplified 2-state glycolysis model:
    y[0] = F6P, y[1] = F16BP
    p = (v, k1, K, n, k2)
    """
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = 1 + (F16BP / K) ** n
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# -------------------------------------------
# 2. Simulation setup: parameters and data
# -------------------------------------------
times = np.arange(0, 15, 0.01)  # time points
params = (14, 1, 2, 3, 4)       # true parameters (v, k1, K, n, k2)

# Generate synthetic clean data by integrating the ODE system
y = odeint(glycolysis_ode, y0=[0, 0], t=times, args=(params,), rtol=1e-8)

# -------------------------------------------
# 3. PySINDy model hyperparameters
# -------------------------------------------
dt = 0.01                # time step size
poly_order = 4           # polynomial degree for library functions
threshold = 0.05         # sparsity threshold for STLSQ optimizer

# Training data fractions to evaluate
train_sizes = np.linspace(0.1, 0.9, 9)  # 0.1 to 0.9 in steps of 0.1

# Store R² scores for training and future predictions
r2_train = []
r2_future = []

# -------------------------------------------
# 4. Loop over different training data sizes
# -------------------------------------------
for frac in train_sizes:
    # Calculate number of training points
    train_len = int(frac * len(y))

    # Initialize PySINDy model with STLSQ optimizer and polynomial features
    model = ps.SINDy(
        optimizer=ps.STLSQ(threshold=threshold),
        feature_library=ps.PolynomialLibrary(degree=poly_order),
    )

    # Fit the model to the training subset
    model.fit(y[:train_len], t=dt)

    # Simulate forward from initial condition over training and future time intervals
    x_sim_train = model.simulate([0, 0], times[:train_len])
    x_sim_future = model.simulate(x_sim_train[-1], times[train_len:])

    # Calculate R² for training and future predictions (F6P only)
    r2_train_val = r2_score(y[:train_len, 0], x_sim_train[:, 0])
    r2_future_val = r2_score(y[train_len:, 0], x_sim_future[:, 0]) if train_len < len(y) else 0

    r2_train.append(r2_train_val)
    r2_future.append(r2_future_val)

    # -------------------------------------------
    # 5. Plot true data and model predictions
    # -------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(times, y[:, 0], 'r', linewidth=3, label='Observed $F6P(t)$')
    plt.plot(times, y[:, 1], 'b', linewidth=3, label='Observed $F16BP(t)$')
    plt.plot(times[train_len:], x_sim_future[:, 0], 'k-', label='SINDy Predicted $F6P(t)$')
    plt.plot(times[train_len:], x_sim_future[:, 1], 'k--', label='SINDy Predicted $F16BP(t)$')
    plt.axvline(x=dt*train_len, color='g', linestyle='--', label='Training Cutoff')
    plt.xlabel('Time [s]')
    plt.ylabel('Concentration [mM]')
    plt.title(f'Training Size = {train_len} points ({frac:.1f} fraction)')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------------------
# 6. Summary: print R² scores for all train sizes
# -------------------------------------------
print("\nR² Scores for Training and Future Predictions:")
for frac, r2_t, r2_f in zip(train_sizes, r2_train, r2_future):
    print(f"Train fraction: {frac:.2f}, R² (Train F6P): {r2_t:.4f}, R² (Future F6P): {r2_f:.4f}")

# -------------------------------------------
# 7. Plot evolution of R² with training size
# -------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, r2_train, 'bo-', label='R² (Training Data)')
plt.plot(train_sizes, r2_future, 'rx-', label='R² (Future Data)')
plt.xlabel('Training Size (fraction of total data)')
plt.ylabel('R² Score (F6P)')
plt.title('SINDy Model Performance vs. Training Data Size')
plt.legend()
plt.grid(True)
plt.show()
