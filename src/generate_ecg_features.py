
import numpy as np
import os
import csv

from utils import read_csv

from settings import MEDIANS_PATH, IMAGE_SHAPE, GROUND_TRUTH_PATH
from settings import MEDIANS_FEATURE_PATH, PREDICTION_LABELS, DATA_LABELS

def generate_ecg_feature(ecg_data, ecg_id, save_path, ground_truth):

    feature_vector = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for row in ecg_data:
        feature_vector.extend(row)

    feature_vector.extend(ground_truth)

    np.savetxt(os.path.join(save_path, f'{ ecg_id }.csv'), feature_vector, fmt='%i', newline=',')

def generate_ecg_vector(ecg_data, ecg_id, save_path, ground_truth):

    feature_vector = [ int(ecg_id) ]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for row in ecg_data:
        feature_vector.extend(row)

    feature_vector.extend(ground_truth)

    return feature_vector

if __name__ == '__main__':

    feature_vectors = []

    ground_truth = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    truth_dict = {}
    prediction_indicies = [ DATA_LABELS[label] for label in PREDICTION_LABELS ]

    for row in ground_truth:
        med_id = row[0]
        truth_dict[med_id] = [ int(row[index]) for index in prediction_indicies ]

    for index, ecg_file in enumerate(os.listdir(MEDIANS_PATH)):
        try:
            print(f'Generating features: { index + 1 } / { len(os.listdir(MEDIANS_PATH)) }', end='\r')
            truth = truth_dict[os.path.splitext(ecg_file)[0]]
            ecg_data = read_csv(os.path.join(MEDIANS_PATH, ecg_file), delimiter=' ', transpose=True, skip_header=False, dtype=np.int)
            feature_vector = generate_ecg_vector(ecg_data, os.path.splitext(ecg_file)[0], MEDIANS_FEATURE_PATH, truth)
            feature_vectors.append(feature_vector)
        except:
            pass

    np.savetxt(os.path.join(MEDIANS_FEATURE_PATH, f'all_patients.csv'), feature_vectors, fmt='%i', delimiter=',')
    
