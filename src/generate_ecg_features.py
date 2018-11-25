
import numpy as np
import os
import csv
import shutil

from utils import read_csv

from settings import MEDIANS_PATH, GROUND_TRUTH_PATH
from settings import MEDIANS_FEATURE_PATH, PREDICTION_LABELS, DATA_LABELS

def generate_ecg_feature(ecg_data, ecg_id, ground_truth, save_path=None):

    feature_vector = [ int(ecg_id) ]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for row in ecg_data:
        feature_vector.extend(row)

    feature_vector.extend(ground_truth)

    if save_path is None:
        np.savetxt(os.path.join(save_path, f'{ ecg_id }.csv'), feature_vector, fmt='%i', newline=',')

    return feature_vector

if __name__ == '__main__':


    if os.path.exists(MEDIANS_FEATURE_PATH):
        shutil.rmtree(MEDIANS_FEATURE_PATH)

    feature_vectors = []

    ground_truth = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    prediction_indicies = [ DATA_LABELS[label] for label in PREDICTION_LABELS ]

    for index, row in enumerate(ground_truth):
        print(f'Generating features: { index + 1 } / { len(ground_truth) }', end='\r')
        
        medians_path = os.path.join(MEDIANS_PATH, f'{ row[0] }.asc')

        if os.path.exists(medians_path):

            # Read ECG data from .asc file
            ecg_data = read_csv(
                medians_path,
                delimiter=' ',
                transpose=True,
                skip_header=False,
                dtype=np.int)
            
            # Generate ECG feature vector
            ecg_feature = generate_ecg_feature(
                ecg_data,
                row[0],
                [ int(row[ index ]) for index in prediction_indicies ],
                MEDIANS_FEATURE_PATH)

            feature_vectors.append(ecg_feature)

    np.savetxt(os.path.join(MEDIANS_FEATURE_PATH, f'all_patients.csv'), feature_vectors, fmt='%i', delimiter=',')

    print('\n', end='\r')