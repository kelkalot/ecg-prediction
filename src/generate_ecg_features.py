
import numpy as np
import os
import csv

from utils import read_csv

from settings import MEDIANS_PATH, IMAGE_SHAPE

def generate_ecg_feature(ecg_data, ecg_id, save_path='../data/medians_features'):

    feature_vector = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for row in ecg_data:
        feature_vector.extend(row)

    np.savetxt(os.path.join(save_path, f'{ ecg_id }.csv'), feature_vector, fmt='%i', newline=',')

if __name__ == '__main__':

    for index, ecg_file in enumerate(os.listdir(MEDIANS_PATH)):
        print(f'Generating features: { index + 1 } / { len(os.listdir(MEDIANS_PATH)) }', end='\r')
        ecg_data = read_csv(os.path.join(MEDIANS_PATH, ecg_file), delimiter=' ', transpose=True, skip_header=False, dtype=np.int)
        generate_ecg_feature(ecg_data, os.path.splitext(ecg_file)[0])
    
