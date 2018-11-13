
import csv
import numpy as np
import tensorflow as tf
try:
    import matplotlib.pyplot as plt
except:
    plt = None

DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCCalculation': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14
}

def plot_loss(history, file_name):

    if plt is None:
        return
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend(['training', 'validation'], loc='upper left')

    plt.savefig(file_name)
    plt.gcf().clear()

def read_csv(csv_file, delimiter=';', shape=None, skip_header=True):

    data = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            next(csv_data)
        
        for row in csv_data:
            data.append(row)

    data = np.array(data)

    if shape is not None and shape != data.shape:
        data = np.reshape(data, shape)

    return data

def prepare_data(data, prediction_labels, training_path, x_shape=None):

    x_data = []
    y_data = []

    for row in data:
        try:
            median_data = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
            indicies = [DATA_LABELS[label] for label in prediction_labels]
            y_data.append([row[index] for index in indicies])
            x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]