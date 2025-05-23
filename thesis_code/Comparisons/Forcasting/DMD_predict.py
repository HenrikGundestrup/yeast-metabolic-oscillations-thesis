import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pydmd import DMD
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
# 2. Generate synthetic data by integrating the ODE
# -------------------------------------------
times = np.arange(0, 15, 0.01)      # Time points from 0 to 15 s in 0.01 s increments
params = (14, 1, 2, 3, 4)           # Model parameters (v, k1, K, n, k2)
y = odeint(glycolysis_ode, [0, 0], times, args=(params,), rtol=1e-8)  # Simulated clean data

# -------------------------------------------
# 3. Delay embedding function for DMD input
# -------------------------------------------
def future_delay_embedding(data, delay_steps):
    """
    Creates delay-embedded data by stacking consecutive time steps.
    
    Parameters:
    - data: original time series array of shape (T, features)
    - delay_steps: number of consecutive steps to stack
    
    Returns:
    - embedded_data: array of shape (T-delay_steps, delay_steps * features)
    """
    embedded_data = []
    for i in range(len(data) - delay_steps):
        # Flatten delay_steps of consecutive vectors into one long vector
        stacked_row = data[i:i + delay_steps].flatten()
        embedded_data.append(stacked_row)
    return np.array(embedded_data)

# -------------------------------------------
# 4. Parameters for delay embedding and training
# -------------------------------------------
delay_steps = 150                     # Number of steps for delay embedding
train_sizes = np.linspace(0.1, 0.9, 9)  # Fractions of training data to evaluate

# Lists to store R² scores
r2_train = []
r2_future = []

# -------------------------------------------
# 5. Loop over different training sizes
# -------------------------------------------
for train_fraction in train_sizes:
    # Optionally add noise here (currently zero noise)
    yobs = np.random.normal(y, 0.0)
    
    # Apply delay embedding
    yobs_embedded = future_delay_embedding(yobs, delay_steps)
    times_embedded = times[:len(times) - delay_steps]

    # Define training and test split indices
    train_size = int(train_fraction * len(yobs_embedded))
    y_train = yobs_embedded[:train_size]
    y_test = yobs_embedded[train_size:]

    # Fit DMD model to training data (transpose because pydmd expects features x snapshots)
    dmd = DMD(svd_rank=30)
    dmd.fit(y_train.T)

    # Extend reconstruction to full embedded timeline
    dmd.dmd_time['tend'] = len(yobs_embedded) - 1
    y_dmd_full = dmd.reconstructed_data.real.T  # Reconstructed data shape: (time, features)

    # Split reconstruction into training and future parts
    y_dmd_train = y_dmd_full[:train_size]
    y_dmd_future = y_dmd_full[train_size:]

    # Extract corresponding true data for evaluation
    true_train = y[:train_size]
    true_future = y[train_size:train_size + len(y_dmd_future)]

    # Calculate R² scores for the first state variable (F6P)
    r2_t = r2_score(true_train[:, 0], y_dmd_train[:, 0])
    r2_f = r2_score(true_future[:, 0], y_dmd_future[:, 0])
    r2_train.append(r2_t)
    r2_future.append(r2_f)

    # -------------------------------------------
    # 6. Plot true vs predicted data for this split
    # -------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(times, y[:, 0], "r", label="$F6P$ True", linewidth=2)
    plt.plot(times, y[:, 1], "b", label="$F16BP$ True", alpha=0.4, linewidth=2)
    plt.plot(times[train_size:train_size + len(y_dmd_future)], y_dmd_future[:, 0], "k--", label="DMD Future $F6P$")
    plt.plot(times[train_size:train_size + len(y_dmd_future)], y_dmd_future[:, 1], "k--", label="DMD Future $F16BP$")
    plt.axvline(x=times[train_size], color='b', linestyle='--', label='Train/Test Split')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mM)')
    plt.title(f'DMD Prediction - Train Size = {train_size} points ({train_fraction:.2f} fraction)')
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------------------------------
# 7. Print R² scores for all training sizes
# -------------------------------------------
print("\nR² Scores for Training and Future Predictions (F6P):")
for frac, r2_t, r2_f in zip(train_sizes, r2_train, r2_future):
    print(f"Train fraction: {frac:.2f}, R² (Train): {r2_t:.4f}, R² (Future): {r2_f:.4f}")

# -------------------------------------------
# 8. Plot evolution of R² scores vs training size
# -------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, r2_train, 'bo-', label='R² (Training)')
plt.plot(train_sizes, r2_future, 'rx-', label='R² (Future)')
plt.xlabel('Training Size (fraction of total data)')
plt.ylabel('R² Score (F6P)')
plt.title('DMD Model Performance vs. Training Data Size')
plt.legend()
plt.grid(True)
plt.show()
