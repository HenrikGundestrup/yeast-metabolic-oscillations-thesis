import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.metrics import r2_score

# -----------------------
# ODE Model: Glycolysis
# -----------------------
def Cell(y, t, p):
    """2-state glycolysis model equations."""
    dn = p[0] - p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0]
    dc = p[1] * (1 + (y[1] / p[2]) ** p[3]) * y[0] - p[4] * y[1]
    return [dn, dc]

# Generate synthetic solution of the system
times = np.arange(0, 15, 0.01)
params = (14, 1, 2, 3, 4)  # (v, k1, K, n, k2)
y = odeint(Cell, [0, 0], times, args=(params,), rtol=1e-8)

# -----------------------
# Sequence Preparation
# -----------------------
def create_sequences(data, seq_length):
    """Split time series into sequences for supervised learning."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# -----------------------
# LSTM Model Definition
# -----------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # use output at last time step

# -----------------------
# Training and Evaluation
# -----------------------
# Hyperparameters
sequence_length = 10
num_epochs = 200
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_sizes = np.linspace(0.1, 0.9, 9)

r2_train = []
r2_future = []

# Loop through different training fractions
for train_fraction in train_sizes:
    # Prepare data
    X, y_target = create_sequences(y, sequence_length)
    train_size = int(train_fraction * len(X))
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X[:train_size], dtype=torch.float32)
    y_train = torch.tensor(y_target[:train_size], dtype=torch.float32)
    X_test = torch.tensor(X[train_size:], dtype=torch.float32)
    y_test = torch.tensor(y_target[train_size:], dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = LSTMModel(input_size=2, hidden_size=64, num_layers=2, output_size=2).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train.to(device)).cpu().numpy()
        pred_test = model(X_test.to(device)).cpu().numpy()

    # Compute R² scores for F6P (first state variable)
    r2_t = r2_score(y_target[:train_size, 0], pred_train[:, 0])
    r2_f = r2_score(y_target[train_size:, 0], pred_test[:, 0])
    r2_train.append(r2_t)
    r2_future.append(r2_f)

    # Plot prediction vs true
    plt.plot(times, y[:, 0], "r", label="F6P True", linewidth=2)
    plt.plot(times, y[:, 1], "b", label="F16BP True", alpha=0.4, linewidth=2)
    pred_time = times[sequence_length + train_size:sequence_length + train_size + len(pred_test)]
    plt.plot(pred_time, pred_test[:, 0], "k--", label="LSTM Future F6P")
    plt.plot(pred_time, pred_test[:, 1], "k--", label="LSTM Future F16BP")
    plt.axvline(x=times[sequence_length + train_size], color='b', linestyle='--', label='Train/Test Split')
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mM)')
    plt.title(f'LSTM Prediction - Train Size = {train_size} points')
    plt.legend()
    plt.show()

# -----------------------
# Summary and Visualization
# -----------------------
for ts, r2t, r2f in zip(train_sizes, r2_train, r2_future):
    print(f"Training size: {ts:.1f}, R² (Train) for F6P: {r2t:.4f}, R² (Future) for F6P: {r2f:.4f}")

# Plot R² scores vs training size
plt.plot(train_sizes, r2_train, label="R² (Training)", marker='o', color='b')
plt.plot(train_sizes, r2_future, label="R² (Future)", marker='x', color='r')
plt.xlabel('Training Size (fraction of total data)')
plt.ylabel('R² Score')
plt.title('LSTM: Evolution of R² Scores with Training Size')
plt.legend()
plt.grid(True)
plt.show()
