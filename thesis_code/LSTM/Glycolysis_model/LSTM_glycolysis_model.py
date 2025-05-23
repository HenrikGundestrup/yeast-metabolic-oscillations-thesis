# ----------------------------------------------------------------------------------
# LSTM on Synthetic 2-State Glycolysis Model
# ----------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------------------
# 1. Define the Glycolysis ODE System (2-state model)
# ----------------------------------------------------------------------------------

def Cell(y, t, p):
    """
    Simple 2-state glycolysis system.
    y[0] = F6P, y[1] = F16BP
    """
    dn = p[0] - p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# ----------------------------------------------------------------------------------
# 2. Generate Synthetic Time Series Data
# ----------------------------------------------------------------------------------

# Time points for integration
times = np.arange(0, 15, 0.01)

# Model parameters: v, k1, K, n, k2
params = (15, 1, 2, 3, 4)

# Solve ODE system
y = odeint(Cell, y0=[0, 0], t=times, args=(params,), rtol=1e-8)

# Add optional Gaussian noise to simulate observations
noise_std = 0.0
yobs = np.random.normal(y, noise_std)

# ----------------------------------------------------------------------------------
# 3. Normalize Data
# ----------------------------------------------------------------------------------

scaler = MinMaxScaler()
yobs_scaled = scaler.fit_transform(yobs)

# ----------------------------------------------------------------------------------
# 4. Prepare Data Using Future Delay Embedding
# ----------------------------------------------------------------------------------

def future_delay_embedding(data, delay_steps):
    """
    Generates input sequences and corresponding target vectors using sliding windows.
    """
    X, Y = [], []
    for i in range(len(data) - delay_steps):
        X.append(data[i:i + delay_steps])
        Y.append(data[i + delay_steps])
    return np.array(X), np.array(Y)

delay_steps = 10
X, Y = future_delay_embedding(yobs_scaled, delay_steps)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# ----------------------------------------------------------------------------------
# 5. Define LSTM Model
# ----------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=50, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)             # out: (batch, seq_len, hidden)
        out = self.fc(out[:, -1, :])      # Take the last time step output
        return out

# Instantiate model, loss, and optimizer
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------------------------------------------------
# 6. Train the Model
# ----------------------------------------------------------------------------------

epochs = 200
for epoch in range(epochs):
    model.train()
    output = model(X_tensor)
    loss = criterion(output, Y_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# ----------------------------------------------------------------------------------
# 7. Predict Using Trained Model
# ----------------------------------------------------------------------------------

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

# Inverse transform the predictions to original scale
# Append dummy zeros to match dimensionality, then slice to true size
y_pred_full = np.hstack([y_pred_scaled, np.zeros((y_pred_scaled.shape[0], yobs.shape[1] - y_pred_scaled.shape[1]))])
y_pred = scaler.inverse_transform(y_pred_full)[:, :2]

# Time vector for predictions (shifted)
times_embedded = times[delay_steps:]

# ----------------------------------------------------------------------------------
# 8. Plot Predictions vs True Dynamics
# ----------------------------------------------------------------------------------

plt.figure(figsize=(10, 5))

# Plot true dynamics
plt.plot(times, y[:, 0], 'r', label='True $F6P(t)$', linewidth=4.0)
plt.plot(times, y[:, 1], 'b', label='True $F16BP(t)$', linewidth=4.0)

# Plot LSTM predictions
plt.plot(times_embedded, y_pred[:, 0], 'k-', label='LSTM $F6P(t)$')
plt.plot(times_embedded, y_pred[:, 1], 'k--', label='LSTM $F16BP(t)$')

plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------
# 9. Evaluate Performance Using R² Score
# ----------------------------------------------------------------------------------

r2_f6p = r2_score(y[delay_steps:, 0], y_pred[:, 0])
r2_f16bp = r2_score(y[delay_steps:, 1], y_pred[:, 1])
r2_avg = (r2_f6p + r2_f16bp) / 2

print(f"R² F6P: {r2_f6p:.4f}")
print(f"R² F16BP: {r2_f16bp:.4f}")
print(f"Average R²: {r2_avg:.4f}")
