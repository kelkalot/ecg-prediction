import numpy as np
import os
import csv
import shutil
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# from scipy.interpolate import spline

from utils import read_csv, normalize, shorten

from settings import MEDIANS_PATH, MEDIANS_PLOT_PATH

PLOT_FORMAT = 'png'
PLOT_COLORS = [
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
    '#000000',
]

PLOT_DPI = 299

Y_MAX = 299
X_MAX = 299

Y_TOP_LIMIT = 3000
Y_BOTTOM_LIMIT = -3000

def generate_ecg_plot(ecg_data, ecg_id, save_path=None):
   
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    plt.figure(figsize=(X_MAX / PLOT_DPI, Y_MAX / PLOT_DPI), dpi=PLOT_DPI)
    plt.axis('off')

    for index, row in enumerate(ecg_data):

        x = range(len(row))
        y = row

        # x_smooth = np.linspace(min(x), max(x), len(x))
        # y_smooth = spline(x, y, x_smooth)
        
        plt.plot(x, y, color=PLOT_COLORS[index], linewidth=0.1)

    plt.ylim(Y_BOTTOM_LIMIT, Y_TOP_LIMIT)

    plt.savefig(
        save_path,
        format=PLOT_FORMAT)

    plt.close()

if __name__ == '__main__':

    median_files = sorted(os.listdir(MEDIANS_PATH))
    
    if os.path.exists(MEDIANS_PLOT_PATH):
        shutil.rmtree(MEDIANS_PLOT_PATH)

    for index, ecg_file in enumerate(median_files):
        print(f'Generating plots: { index + 1 } / { len(median_files) }', end='\r')

        ecg_id = os.path.splitext(ecg_file)[0]

        file_path = os.path.join(MEDIANS_PLOT_PATH, f'{ ecg_id }.{ PLOT_FORMAT }')

        # Read ECG data from .asc file
        ecg_data = read_csv(
            os.path.join(MEDIANS_PATH, ecg_file),
            delimiter=' ',
            transpose=True,
            skip_header=False,
            dtype=np.int)

        # Normalize ECG data between -1 and 1
        ecg_normalized = normalize(ecg_data)

        # Scale normlaized ECG data between -Y_MAX and Y_MAX
        ecg_scaled = np.array(ecg_normalized * Y_MAX, dtype=int)

        # Reduce the length of ECG data by dropping every other element
        ecg_reduced = shorten(ecg_scaled)

        # Remove the first n values to fit within the X_MAX limit
        if len(ecg_reduced[0]) > X_MAX:
            ecg_reduced = [ ecg_values[len(ecg_values) - X_MAX:] for ecg_values in ecg_reduced ]

        # Generate ECG plot
        generate_ecg_plot(ecg_data, ecg_id, file_path)

        plot_image = Image.open(file_path).convert('L')
        plot_image = np.array(plot_image)

        # plot_image = plot_image[~np.all(plot_image == 255, axis=1)]
        plot_image = plot_image.compress(~np.all(plot_image == 255, axis=0), axis=1)
        plot_image = Image.fromarray(plot_image, 'L')

        plot_image.save(file_path)
    
    print('\n', end='\r')