import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from skimage import io
import os
from glob import glob

# Directory containing experimental TIFF files and corresponding mask PNGs
directory = 'C:\\Users\\henri\\OneDrive\\Skrivebord\\UNI\\9. Semester\\Cell_Data\\Data'

# --------------------------------------------------------------------
# STEP 1: Extract per-cell intensity time series from TIFF + mask
# --------------------------------------------------------------------
for epi_file in glob(f'{directory}/*/*/blue_kinetic_+glucose_after30_02.tif'):
    img_epi = io.imread(epi_file)  # Time-lapse fluorescence image stack
    path = os.path.dirname(os.path.abspath(epi_file))
    filename = os.path.basename(path)
    print(f"Processing: {filename}")

    # Find and load the cell mask (.png)
    for mask_file in os.listdir(path):
        if mask_file.endswith('.png'):
            img_mask = io.imread(os.path.join(path, mask_file))

    # Compute mean intensity per cell over time and save as .npy
    output_path = os.path.join(directory, path + '.npy')
    with open(output_path, "wb") as f:
        i_max = img_mask.max()  # Number of cells
        arr = np.zeros((i_max, len(img_epi)))  # (cells × time) matrix

        for i in range(1, i_max + 1):
            mask = (img_mask == i)
            area = np.sum(mask)
            for t in range(len(img_epi)):
                intensity = np.sum(img_epi[t] * mask)
                arr[i - 1, t] = intensity / area  # Mean intensity

        np.save(f, arr)

# --------------------------------------------------------------------
# STEP 2: Load all .npy intensity files
# --------------------------------------------------------------------
all_data = []
for file in os.listdir(directory):
    if file.endswith('.npy'):
        data = np.load(os.path.join(directory, file))
        all_data.append(data)

# Compute mean signal across all cells from the first file
mean = all_data[0].mean(axis=0)

# --------------------------------------------------------------------
# STEP 3: Define delay embedding function
# --------------------------------------------------------------------
def future_delay_embedding(data, delay_steps):
    """
    Construct delay-embedded dataset from 1D time series using future time steps.
    """
    embedded_data = []
    for i in range(len(data) - delay_steps):
        window = data[i:i + delay_steps].flatten()
        embedded_data.append(window)
    return np.array(embedded_data)

# --------------------------------------------------------------------
# STEP 4: Run DMD with varying training sizes and delays, and visualize
# --------------------------------------------------------------------
# Time vector (assumed 1 second resolution, 600 seconds total)
times = np.linspace(0, 600, 600)

# Experiment configs
data_points_list = [100, 300, 500]        # Training sizes
delay_ratios = [0.25, 0.50, 0.75]         # Delay as fraction of training points

# Set up subplot grid
fig, axes = plt.subplots(len(data_points_list), len(delay_ratios), figsize=(15, 12))

# Loop through each configuration
for i, data_points in enumerate(data_points_list):
    model_data = mean[:data_points]

    for j, delay_ratio in enumerate(delay_ratios):
        delay_steps = int(data_points * delay_ratio)
        yobs_embedded = future_delay_embedding(model_data, delay_steps)

        # Fit DMD to embedded data
        dmd = DMD(svd_rank=20)
        dmd.fit(yobs_embedded.T)

        # Extrapolate DMD solution over full time range
        dmd.dmd_time['dt'] = 1
        dmd.dmd_time['tend'] = 600
        y_dmd_embedded = dmd.reconstructed_data.real
        y_dmd_signal = y_dmd_embedded[0, :-1]  # First delay component

        # Plot
        ax = axes[i, j]
        ax.plot(times, mean, 'b', label='Original Data')
        ax.plot(times, y_dmd_signal, color='lightgreen', linestyle='--', label='DMD Model')
        ax.axvline(x=data_points, color='g', linestyle='--', label='Cutoff')

        # Titles and labels
        ax.set_title(f"{data_points} pts, Delay {int(delay_ratio * 100)}%")
        if j == 0:
            ax.set_ylabel('Concentration (a.u.)')
        else:
            ax.set_yticklabels([])
        if i == len(data_points_list) - 1:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])

        ax.grid()

        # Show legend on only one plot
        if i == 0 and j == 0:
            ax.legend()
        else:
            ax.legend().set_visible(False)

# Rotate x-tick labels for readability
for ax in axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(45)

# Final layout adjustments
fig.tight_layout()
fig.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
plt.show()
