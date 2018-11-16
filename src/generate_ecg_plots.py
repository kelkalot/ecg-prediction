import numpy as np
import os
import csv

import matplotlib.pyplot as plt

from utils import read_csv

from settings import MEDIANS_PATH, MEDIANS_PLOT_PATH

PLOT_COLORS = [
    '#5ea3a3',
    '#fa0559',
    '#c8e4fe',
    '#5c848e',
    '#ff9234',
    '#fd0054',
    '#f9ad8d',
    '#80ac7b',
]

def generate_ecg_plot(ecg_data, ecg_id, save_path=None):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for index, row in enumerate(ecg_data):
        plt.plot(range(len(row)), row, color=PLOT_COLORS[index])

    plt.axis('off')
    plt.tick_params(axis='x', which='major')
    plt.savefig(os.path.join(save_path, f'{ ecg_id }.pdf'), format='pdf')
    plt.clf()

if __name__ == '__main__':

    median_files = os.listdir(MEDIANS_PATH)

    for index, ecg_file in enumerate(median_files):
        print(f'Generating features: { index + 1 } / { len(median_files) }', end='\r')
        ecg_data = read_csv(os.path.join(MEDIANS_PATH, ecg_file), delimiter=' ', transpose=True, skip_header=False, dtype=np.int)
        generate_ecg_plot(ecg_data, os.path.splitext(ecg_file)[0], MEDIANS_PLOT_PATH)
    
    print('\n', end='\r')