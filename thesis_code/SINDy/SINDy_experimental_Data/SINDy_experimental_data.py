import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary

# --- Load .npy files ---
directory = 'C:\\Users\\henri\\OneDrive\\Skrivebord\\UNI\\9. Semester\\Cell_Data\\Data'
all_data = []

for file in os.listdir(directory):
    if file.endswith(".npy"):
        data = np.load(os.path.join(directory, file))
        all_data.append(data)

# --- Use mean signal of 15 mM glucose condition ---
mean_signal = np.nanmean(all_data[0], axis=0)
time = np.arange(len(mean_signal))
dt = 1.0  # 1 second interval

# --- Optional detrending ---
# from scipy.signal import detrend
# mean_signal = detrend(mean_signal)

# --- Cut data for training ---
data_cut = 600
x_train = mean_signal[:data_cut].reshape(-1, 1)
t_train = time[:data_cut]

# --- Normalize the training data ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# --- Fit SINDy model ---
model = SINDy(
    optimizer=STLSQ(threshold=0.05),
    feature_library=PolynomialLibrary(degree=7),
    discrete_time=True
)
model.fit(x_train_scaled)
model.print()

# --- Simulate forward and inverse transform ---
x_sim_scaled = model.simulate(x_train_scaled[0], len(mean_signal))
x_sim = scaler.inverse_transform(x_sim_scaled)

# --- Performance metrics ---
r2 = r2_score(mean_signal, x_sim)
rmse = np.sqrt(mean_squared_error(mean_signal, x_sim))
print(f"R2 score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- Plot results ---
plt.figure(figsize=(10, 4))
plt.plot(time, mean_signal, 'b', label="Original Data")
plt.plot(time, x_sim, '--k', label="SINDy Prediction")
plt.axvline(x=data_cut, color='gray', linestyle='--', label='Training Cutoff')
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")
plt.title("SINDy Fit on Glycolytic Oscillations (15 mM Glucose)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
