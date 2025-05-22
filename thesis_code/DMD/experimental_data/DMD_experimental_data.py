import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import os
from glob import glob
from pydmd import DMD
from sklearn.metrics import r2_score

directory = 'C:\\Users\\henri\\OneDrive\\Skrivebord\\UNI\\9. Semester\\Cell_Data\\Data'

# Generate .npy files with cell-wise intensity data
for epi_file in glob(f'{directory}/*/*/blue_kinetic_+glucose_after30_02.tif'):
    img_epi = io.imread(epi_file)
    path = os.path.dirname(os.path.abspath(epi_file))
    filename = os.path.basename(path)
    print(filename)
    for mask_file in os.listdir(path):
        if '.png' in mask_file:
            img_mask = io.imread(os.path.join(path, mask_file))
    with open(os.path.join(directory, path + '.npy'), "wb") as f:
        i_max = img_mask.max()
        arr = np.zeros((i_max, len(img_epi)))
        for i in range(1, i_max + 1):
            mask = (img_mask == i)
            masked = mask * img_epi
            area = np.sum(mask)
            for j in range(len(masked)):
                intdens = np.sum(masked[j])
                arr[i-1, j] = intdens / area
        np.save(f, arr)

# Load all .npy files
all_data = []
for files in os.listdir(directory):
    if '.npy' in files:
        data = np.load(os.path.join(directory, files))
        all_data.append(data)

# Plot time series
fig, ax = plt.subplots(5, 1, figsize=(8, 20))
fig.suptitle('Glycolytic Oscillations per Cell', fontsize=16)
titles = ['All cells, 15 mM glucose', 'All cells, 30 mM glucose', 'All cells, 3 mM glucose', 'All cells, 7.5 mM glucose']
for j in range(4):
    ax[j].set_title(titles[j])
    for i in range(len(all_data[j])):
        ax[j].plot(all_data[j][i, :])
mean = all_data[0].mean(axis=0)
ax[4].set_title('Mean')
ax[4].plot(mean)
ax[4].set_xlim([0, 600])
ax[4].legend(['15 mM glucose', '30 mM glucose', '3 mM glucose', '7.5 mM glucose'])
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.savefig(directory + "/Glycolytic_oscillations_per_cell.svg", dpi=1200)
plt.show()
plt.close()

# DMD data preparation
sum_15mM = np.sum(all_data[0], axis=0) / len(all_data[0])
sum_30mM = np.sum(all_data[1], axis=0)
times = np.arange(0, 600, 1)
mean = all_data[0].mean(axis=0)

data_cut = 600
model_data = mean[:data_cut]
model_time = times[:data_cut]

plt.plot(model_time, model_data)
plt.title('Data the DMD model gets')
plt.show()
plt.close()

# Delay embedding function
def future_delay_embedding(data, delay_steps):
    return np.array([data[i:i + delay_steps].flatten() for i in range(len(data) - delay_steps)])

delay_steps = 200
yobs_embedded = future_delay_embedding(model_data, delay_steps)

# Fit DMD
dmd = DMD(svd_rank=20)
dmd.fit(yobs_embedded.T)
dmd.dmd_time['dt'] = 1
dmd.dmd_time['tend'] = 600

# Reconstruct and align
y_dmd_embedded = dmd.reconstructed_data.real
y_dmd_embedded = y_dmd_embedded[0][:-1]

# Plot DMD vs original
plt.plot(times, mean, 'b', label='Original Data', linewidth=3)
plt.plot(times[:600], y_dmd_embedded, '--', color='lightgreen', label='DMD Model')
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mM)')
plt.legend()
plt.grid()
plt.show()

# R² Score
r2 = r2_score(y_dmd_embedded, mean[0:len(y_dmd_embedded)])
print(r2)
