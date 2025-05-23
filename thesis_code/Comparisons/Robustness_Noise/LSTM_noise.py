# LSTM Robustness Test on Noisy 2-State Glycolysis System

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score

# ----------------------------------------
# 1. Define the Glycolysis ODE Model
# ----------------------------------------
def glycolysis_ode(y, t, p):
    F6P, F16BP = y
    v, k1, K, n, k2 = p
    term = 1 + (F16BP / K) ** n
    dF6P_dt = v - k1 * term * F6P
    dF16BP_dt = k1 * term * F6P - k2 * F16BP
    return [dF6P_dt, dF16BP_dt]

# Time vector and model parameters
t = np.arange(0, 15, 0.01)
params = (15, 1, 2, 3, 4)

# Generate clean reference trajectory
y_clean = odeint(glycolysis_ode, y0=[0, 0], t=t, args=(params,), rtol=1e-8)

# ----------------------------------------
# 2. Helper Functions for Sequence Creation and Noise Addition
# ----------------------------------------
def create_sequences(data, seq_length):
    """
    Generates input-output pairs using a sliding window approach.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def add_noise(data, noise_level):
    """
    Adds Gaussian noise to the dataset.
    """
    return data + np.random.normal(0, noise_level, data.shape)

# Create sequences (1 time step input)
seq_len = 1
X, y_target = create_sequences(y_clean, seq_len)

# ----------------------------------------
# 3. Define LSTM Model
# ----------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # Use last hidden state output

# ----------------------------------------
# 4. Noise Robustness Experiment
# ----------------------------------------
noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
r2_results = []

device = 'cuda' if torch.cuda.is_available() else 'cpu'

for noise_level in noise_levels:
    print(f"\nTraining with noise level: {noise_level}")
    
    # Add noise to targets
    noisy_targets = add_noise(y_target, noise_level)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(noisy_targets, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Instantiate model
    model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train model
    for epoch in range(200):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        preds = model(X_tensor.to(device)).cpu().numpy()

    # Compute R² scores
    r2_f6p = r2_score(noisy_targets[:, 0], preds[:, 0])
    r2_f16bp = r2_score(noisy_targets[:, 1], preds[:, 1])
    r2_results.append((noise_level, r2_f6p, r2_f16bp))

    print(f"R² F6P: {r2_f6p:.4f}, R² F16BP: {r2_f16bp:.4f}")

# ----------------------------------------
# 5. Plot R² Scores vs Noise Level
# ----------------------------------------
noise_vals, r2_f6p_vals, r2_f16bp_vals = zip(*r2_results)
r2_avg = (np.array(r2_f6p_vals) + np.array(r2_f16bp_vals)) / 2

plt.plot(noise_vals, r2_avg, marker='o', linestyle='-', color='b', label='Average R²')
plt.xlabel('Noise Standard Deviation')
plt.ylabel('R² Score')
plt.legend()
plt.grid(True)
plt.title('LSTM Performance vs Noise Level')
plt.show()

# ----------------------------------------
# 6. Display Final R² Scores
# ----------------------------------------
print("\nFinal R² Scores by Noise Level:")
for nl, r2_0, r2_1 in r2_results:
    print(f"Noise {nl:.2f} | R² F6P: {r2_0:.4f}, R² F16BP: {r2_1:.4f}")
