import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from pydmd import DMD
from skimage import io
import os
from glob import glob

# Directory containing image and mask data
directory = 'C:\\Users\\henri\\OneDrive\\Skrivebord\\UNI\\9. Semester\\Cell_Data\\Data'

# Extract cell intensity data from .tif and .png files and save as .npy
for epi_file in glob(f'{directory}/*/*/blue_kinetic_+glucose_after30_02.tif'):
    img_epi = io.imread(epi_file)
    path = os.path.dirname(os.path.abspath(epi_file))
    filename = os.path.basename(path)
    print(f"Processing: {filename}")

    # Find corresponding mask image
    for mask_file in os.listdir(path):
        if mask_file.endswith('.png'):
            img_mask = io.imread(os.path.join(path, mask_file))

    # Calculate intensity per cell over time and save
    output_path = os.path.join(directory, path + '.npy')
    with open(output_path, "wb") as f:
        i_max = img_mask.max()
        arr = np.zeros((i_max, len(img_epi)))

        for i in range(1, i_max + 1):
            mask = (img_mask == i)
            masked = mask * img_epi
            area = np.sum(mask)

            for j in range(len(masked)):
                intdens = np.sum(masked[j])
                arr[i - 1, j] = intdens / area  # Mean intensity per cell

        np.save(f, arr)

# Load all .npy files into a list
all_data = []
for file in os.listdir(directory):
    if file.endswith('.npy'):
        data = np.load(os.path.join(directory, file))
        all_data.append(data)

# Calculate mean signal across all cells for one condition
mean = all_data[0].mean(axis=0)

# Function to create future-delay embedding of time series data
def future_delay_embedding(data, delay_steps):
    embedded_data = []
    for i in range(len(data) - delay_steps):
        stacked_row = data[i:i + delay_steps].flatten()
        embedded_data.append(stacked_row)
    return np.array(embedded_data)

# Simulated time axis
times = np.linspace(0, 600, 600)

# Experiment configurations
data_points_list = [200, 300, 400, 500]
delay_ratios = [0.25, 0.50, 0.75]

# Arrays to store R² scores
r2_values = np.zeros((len(data_points_list), len(delay_ratios)))      # R² on training segment
r2_values_hel = np.zeros((len(data_points_list), len(delay_ratios)))  # R² on full 600s signal

# Apply DMD for each configuration
for i, data_points in enumerate(data_points_list):
    model_data = mean[:data_points]  # Truncate data
    model_time = times[:data_points]

    for j, delay_ratio in enumerate(delay_ratios):
        delay_steps = int(data_points * delay_ratio)
        yobs_embedded = future_delay_embedding(model_data, delay_steps)

        # Fit DMD to embedded data
        dmd = DMD(svd_rank=20)
        dmd.fit(yobs_embedded.T)

        # Extrapolate prediction over full time horizon
        dmd.dmd_time['dt'] = 1
        dmd.dmd_time['tend'] = 600

        # Reconstruct signal from DMD and extract first component
        y_dmd_embedded = dmd.reconstructed_data.real
        y_dmd_embedded = y_dmd_embedded[0, :-1]

        # R² scores
        r2_values_hel[i, j] = r2_score(mean, y_dmd_embedded)                         # Full range
        r2_values[i, j] = r2_score(mean[:data_points], y_dmd_embedded[:data_points]) # Training range

# Plotting results
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

# Plot R² on full time horizon
ax = axes[0]
for i, data_points in enumerate(data_points_list):
    ax.plot(
        [25, 50, 75],
        r2_values_hel[i],
        marker='o',
        label=f"{data_points} data points"
    )
ax.set_xlabel("Delay Embedding Ratio (%)")
ax.set_ylabel("$R^2$ Value")
ax.set_title("DMD Performance on Full Signal")
ax.legend()
ax.grid(True)

# Plot R² on training segment only
ax = axes[1]
for i, data_points in enumerate(data_points_list):
    ax.plot(
        [25, 50, 75],
        r2_values[i],
        marker='o',
        label=f"{data_points} data points"
    )
ax.set_xlabel("Delay Embedding Ratio (%)")
ax.set_ylabel("$R^2$ Value")
ax.set_title("DMD Performance on Training Segment")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()
