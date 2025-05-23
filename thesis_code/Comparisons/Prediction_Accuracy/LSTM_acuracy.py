# LSTM Model for Predicting 2-State Glycolysis Dynamics

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score

# ----------------------------------------
# 1. Define ODE System (Simplified Glycolysis Model)
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
params = (14, 1, 2, 3, 4)  # v, k1, K, n, k2

# Simulate the system using ODE solver
y_true = odeint(glycolysis_ode, y0=[0, 0], t=t, args=(params,), rtol=1e-8)

# ----------------------------------------
# 2. Prepare Data for LSTM
# ----------------------------------------
def create_sequences(data, seq_length):
    """
    Constructs input sequences and targets for LSTM.
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Sequence length for LSTM
sequence_length = 10

# Create sequences and convert to tensors
X_seq, y_seq = create_sequences(y_true, sequence_length)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)
y_tensor = torch.tensor(y_seq, dtype=torch.float32)

# Create DataLoader
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

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
        return self.fc(out[:, -1, :])  # Use output of last time step

# Initialize model
model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------------------
# 4. Train the LSTM Model
# ----------------------------------------
num_epochs = 200
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----------------------------------------
# 5. Evaluate Model Performance
# ----------------------------------------
model.eval()
with torch.no_grad():
    predictions = model(X_tensor.to(device)).cpu().numpy()

# Compute R² scores
r2_f6p = r2_score(y_seq[:, 0], predictions[:, 0])
r2_f16bp = r2_score(y_seq[:, 1], predictions[:, 1])
print(f"R² score for F6P: {r2_f6p:.4f}")
print(f"R² score for F16BP: {r2_f16bp:.4f}")

# ----------------------------------------
# 6. Plot Predictions vs True Data
# ----------------------------------------
plt.plot(t[sequence_length:], y_seq[:, 0], 'r', label='Observed $F6P(t)$', linewidth=4)
plt.plot(t[sequence_length:], predictions[:, 0], 'k-', label='LSTM model $F6P(t)$')
plt.plot(t[sequence_length:], y_seq[:, 1], 'b', label='Observed $F16BP(t)$', linewidth=4)
plt.plot(t[sequence_length:], predictions[:, 1], 'k--', label='LSTM model $F16BP(t)$')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.title('LSTM Model Predictions vs True Glycolysis Data')
plt.show()
