import numpy as np
import matplotlib.pyplot as plt
import os
from pysindy import SINDy
from pysindy.optimizers import STLSQ
from pysindy.feature_library import PolynomialLibrary, FourierLibrary, GeneralizedLibrary
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- Load .npy files ---
directory = 'C:\\Users\\henri\\OneDrive\\Skrivebord\\UNI\\9. Semester\\Cell_Data\\Data'
all_data = []

for file in os.listdir(directory):
    if file.endswith(".npy"):
        data = np.load(os.path.join(directory, file))
        all_data.append(data)

# --- Use mean signal from first condition (15 mM glucose) ---
mean_signal = np.nanmean(all_data[0], axis=0)
time = np.arange(len(mean_signal))
dt = 1.0  # 1 second sampling interval

# --- Cut data for training ---
data_cut = 600
x_train = mean_signal[:data_cut].reshape(-1, 1)
t_train = time[:data_cut]

# --- Normalize data ---
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# --- Combined feature library (Polynomial + Fourier) ---
poly_library = PolynomialLibrary(degree=7)
fourier_library = FourierLibrary(n_frequencies=5)
combined_library = GeneralizedLibrary([poly_library, fourier_library])

# --- Fit SINDy model ---
model = SINDy(
    optimizer=STLSQ(threshold=0.05),
    feature_library=combined_library,
    discrete_time=True
)
model.fit(x_train_scaled)
model.print()

# --- Simulate forward and inverse transform ---
x_sim_scaled = model.simulate(x_train_scaled[0], len(mean_signal))
x_sim = scaler.inverse_transform(x_sim_scaled)

# --- Evaluate model ---
r2 = r2_score(mean_signal, x_sim)
rmse = np.sqrt(mean_squared_error(mean_signal, x_sim))
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- Plot results ---
plt.figure(figsize=(10, 4))
plt.plot(time, mean_signal, 'b', label="Original Data")
plt.plot(time, x_sim, '--k', label="SINDy Prediction")
plt.axvline(x=data_cut, color='gray', linestyle='--', label='Training Cutoff')
plt.xlabel("Time (s)")
plt.ylabel("Concentration (mM)")
plt.title("SINDy with Combined Library (Polynomial + Fourier)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
