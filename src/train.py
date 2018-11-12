import keras.backend as K

from keras.optimizers import Nadam

from model import ECGModel
from utils import plot_loss

import csv
import numpy as np
import tensorflow as tf

DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCCalculation': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14
}

PREDICTION_LABELS = [
    'QRSCount', 'QOnset', 'QOffset',
    'POnset', 'POffset', 'TOffset'
]

EPOCHS = 1
BATCH_SIZE = 8
SEED = 2

GROUND_TRUTH_PATH = '/Users/stevenah/github/ecg-prediction/data/ground_truth.csv'
MEDIANS_PATH = '/Users/stevenah/github/ecg-prediction/data/medians'
RHYTHM_PATH = '/Users/stevenah/github/ecg-prediction/data/rhythm'

LOSS_FUNCTION = 'mean_squared_error'
MODEL_FILE = '/Users/stevenah/github/ecg-prediction/model.h5'
PLOT_FILE = 'loss_plot.png'

tf.set_random_seed(SEED)
np.random.seed(SEED)

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]

def read_csv(csv_file, delimiter=',', skip_header=True, as_type='np_array'):

    data = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            next(csv_data)
        
        for row in csv_data:
            data.append(row)

    if as_type == 'np_array':
        data = np.array(data)

    return data

def prepare_data(data, labels=['QOnset']):

    x_data = []
    y_data = []

    for row in data:
        try:
            median_data = read_csv(f'{ MEDIANS_PATH }/{ row[0] }.asc', ' ')
            y_data.append([DATA_LABELS[label] for label in labels])
            x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

if __name__ == '__main__':

    data = read_csv(GROUND_TRUTH_PATH, ';')

    x_data, y_data = prepare_data(data, PREDICTION_LABELS)

    x_train, x_test = split_data(x_data)
    y_train, y_test = split_data(y_data)

    model = ECGModel(output_size=len(PREDICTION_LABELS))

    optimizer = Nadam(lr=0.002)
    
    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    history = model.fit(x_train, y_train, 
          epochs=EPOCHS, 
          batch_size= BATCH_SIZE, 
          verbose=1, 
          validation_data=(x_test, y_test),
          shuffle=True)

    plot_loss(history, PLOT_FILE)

    model.save(MODEL_FILE)



    
    