import csv
import numpy as np

import keras.backend as K
from model import ECGModel
from keras.optimizers import Nadam

DATA_LABELS = {
    'TestID': 0,
    'VentricularRate': 1,
    'P_RInterval': 2,
    'QRSDuration': 3,
    'Q_TInterval': 4,
    'QTCCalculation': 5,
    'PAxis': 6,
    'RAxis': 7,
    'TAxis': 8,
    'QRSCount': 9,
    'QOnset': 10,
    'QOffset': 11,
    'POnset': 12,
    'POffset': 13,
    'TOffset': 14
}

EPOCHS = 1000
BATCH_SIZE = 8



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
            median_data = read_csv(f'./medians/{ row[0] }.asc', ' ')
            y_data.append([DATA_LABELS[label] for label in labels])
            x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]

if __name__ == '__main__':

    data = read_csv('./ground_truth.csv', ';')

    prediction_labels = [
        'QRSCount',
        'QOnset',
        'QOffset',
        'POnset',
        'POffset',
        'TOffset'
    ]

    x_data, y_data = prepare_data(data, prediction_labels)

    x_train, x_test = split_data(x_data)
    y_train, y_test = split_data(y_data)

    model = ECGModel(output_size=len(prediction_labels))

    optimizer = Nadam(lr=0.002)
    
    model.compile(optimizer=optimizer, 
              loss='mean_squared_error',
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
          epochs = EPOCHS, 
          batch_size = BATCH_SIZE, 
          verbose = 1, 
          validation_data = (x_test, y_test),
          shuffle=True)

    model.save('model.h5')



    
    