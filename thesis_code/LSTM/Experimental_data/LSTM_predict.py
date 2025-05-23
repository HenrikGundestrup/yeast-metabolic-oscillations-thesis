# ----------------------------------------------------------------------------------
# LSTM Forecasting on Experimental Cell Intensity Data (Single Cell Mean)
# ----------------------------------------------------------------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from skimage import io
import matplotlib.pyplot as plt
from glob import glob

# ----------------------------------------------------------------------------------
# 1. Define the LSTM Model
# ----------------------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))       # LSTM forward pass
        out = self.fc(out[:, -1, :])          # Take output at last time step
        return out

# ----------------------------------------------------------------------------------
# 2. Preprocess and Save Intensity Traces from TIFF Files
# ----------------------------------------------------------------------------------

# Directory containing .tif and .png data
directory = r'C:\Users\henri\OneDrive\Skrivebord\UNI\9. Semester\Cell_Data\Data'

# Loop through TIFF stacks
for epi_file in glob(f'{directory}/*/*/blue_kinetic_+glucose_after30_02.tif'):
    img_epi = io.imread(epi_file)
    path = os.path.dirname(os.path.abspath(epi_file))
    filename = os.path.basename(path)
    print(filename)

    for mask_file in os.listdir(path):
        if mask_file.endswith('.png'):
            img_mask = io.imread(os.path.join(path, mask_file))

    # Save single-cell traces into .npy
    save_path = os.path.join(directory, filename + '.npy')
    with open(save_path, "wb") as f:
        i_max = img_mask.max()
        arr = np.zeros((i_max, len(img_epi)))
        for i in range(1, i_max + 1):
            mask = (img_mask == i)
            masked = mask * img_epi
            area = np.sum(mask)
            for j in range(len(masked)):
                intdens = np.sum(masked[j])
                arr[i - 1, j] = intdens / area
        np.save(f, arr)

# ----------------------------------------------------------------------------------
# 3. Load Processed Data
# ----------------------------------------------------------------------------------

all_data = []
for file in os.listdir(directory):
    if file.endswith('.npy'):
        data = np.load(os.path.join(directory, file))
        all_data.append(data)

# ----------------------------------------------------------------------------------
# 4. Prepare Data for LSTM Model
# ----------------------------------------------------------------------------------

# Use the mean signal across all cells for a given condition
data = all_data[0].mean(axis=0)

# Define cutoff for training
data_len = 100
sequence_length = 10
data_model = data[:data_len]

# Normalize data to [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
data_model = scaler.fit_transform(data_model.reshape(-1, 1))

# Create input-output sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data_model, sequence_length)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Wrap in DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# ----------------------------------------------------------------------------------
# 5. Train the LSTM Model
# ----------------------------------------------------------------------------------

# Model configuration
input_size = 1
hidden_size = 100
num_layers = 1
output_size = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
model.train()
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# ----------------------------------------------------------------------------------
# 6. Forecast Future Values Using Recursive Prediction
# ----------------------------------------------------------------------------------

future_steps = 600 - data_len     # Total steps to forecast beyond training data
seed_sequence = data_model[-sequence_length:]  # Last observed input sequence
predictions_future = []

model.eval()
with torch.no_grad():
    current_input = torch.tensor(seed_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    for _ in range(future_steps):
        next_step = model(current_input)           # Predict next step
        predictions_future.append(next_step.item())

        next_step = next_step.unsqueeze(-1)        # Add feature dimension
        current_input = torch.cat((current_input[:, 1:, :], next_step), dim=1)

# Denormalize predictions
predictions_future = scaler.inverse_transform(np.array(predictions_future).reshape(-1, 1))

# ----------------------------------------------------------------------------------
# 7. Plot Predictions vs Ground Truth
# ----------------------------------------------------------------------------------

plt.plot(range(len(data)), data, label="True Data", color="blue", linewidth=3.0)
plt.plot(range(data_len, data_len + future_steps), predictions_future, 'k--', label="LSTM Prediction")
plt.axvline(x=data_len, color='g', linestyle='--', label='Training Cutoff')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
