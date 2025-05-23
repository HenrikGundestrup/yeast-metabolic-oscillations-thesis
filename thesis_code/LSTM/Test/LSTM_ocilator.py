import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------------
# 1. Simulate Harmonic Oscillator: dx/dt = v, dv/dt = -x
# ----------------------------------------------------------------------------------

def harmonic_oscillator(t, y):
    return [y[1], -y[0]]  # Standard harmonic oscillator equations

# Time setup
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)
y0 = [1.0, 0.0]  # Initial conditions: x=1, v=0

# Solve ODE
solution = solve_ivp(harmonic_oscillator, t_span, y0, t_eval=t_eval)
t = solution.t
y = solution.y.T  # Shape: (time, 2)

# Add Gaussian noise to simulate real measurements
noise_strength = 0.05
noisy_y = y + noise_strength * np.random.randn(*y.shape)

# ----------------------------------------------------------------------------------
# 2. Normalize Data and Create Sequences for LSTM
# ----------------------------------------------------------------------------------

# Scale to [0, 1] using MinMaxScaler
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(noisy_y)

def create_sequences(data, seq_len):
    """
    Converts a time series into input-output pairs of sequences for supervised learning.
    Each X[i] is a sequence of `seq_len` timesteps, and y[i] is the next step.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

seq_len = 20
X, y_seq = create_sequences(data_scaled, seq_len)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# Create DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ----------------------------------------------------------------------------------
# 3. Define LSTM Model
# ----------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])      # Use last output for prediction
        return out

# Instantiate model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)

# ----------------------------------------------------------------------------------
# 4. Train the Model
# ----------------------------------------------------------------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(100):
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = criterion(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# ----------------------------------------------------------------------------------
# 5. Make Predictions on the Whole Dataset
# ----------------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    X_all = torch.tensor(X, dtype=torch.float32).to(device)
    y_pred_scaled = model(X_all).cpu().numpy()

# Inverse transform to get back original scale
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_seq)

# ----------------------------------------------------------------------------------
# 6. Plot Results
# ----------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

# Plot position
plt.plot(t, noisy_y[:, 0], 'r', linewidth=3, label="True Position (noisy)")
plt.plot(t[seq_len:], y_pred[:, 0], 'k-', label="LSTM Predicted Position")

# Plot velocity
plt.plot(t, noisy_y[:, 1], 'b', linewidth=3, label="True Velocity (noisy)")
plt.plot(t[seq_len:], y_pred[:, 1], 'k--', label="LSTM Predicted Velocity")

plt.xlabel("Time (s)")
plt.ylabel("Position / Velocity")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------
# 7. Evaluate Prediction Performance with R² Score
# ----------------------------------------------------------------------------------

r2_x = r2_score(y[seq_len:, 0], y_pred[:, 0])  # Position
r2_v = r2_score(y[seq_len:, 1], y_pred[:, 1])  # Velocity

print(f"R² Position: {r2_x:.4f}")
print(f"R² Velocity: {r2_v:.4f}")
