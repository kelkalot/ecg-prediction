
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

def read_csv(csv_file, delimiter=';', transpose=False, skip_header=True):

    data = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            next(csv_data)
        
        for row in csv_data:
            data.append(row)

    data = np.array(data, dtype=np.int)

    if transpose:
        data = np.transpose(data)

    return data

def prepare_data(data, prediction_labels, training_path, x_shape=None):

    x_data = []
    y_data = []

    prediction_indicies = [ DATA_LABELS[label] for label in prediction_labels ]

    for row in data:
        try:
            median_data = read_csv(f'{ training_path }/{ row[0] }.asc', ' ', x_shape, False)
            y_data.append([ row[index] for index in prediction_indicies ])
            x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]

def k_fold(x_data, y_data, k=3):

    assert len(x_data) == len(y_data)
    
    x_split_length = len(x_data) // k
    y_split_length = len(y_data) // k

    x_folds = []
    y_folds = []

    for k_index in range(k - 1):
        x_folds += [ x_data[ k_index * x_split_length : (k_index + 1) * x_split_length ] ]
        y_folds += [ y_data[ k_index * y_split_length : (k_index + 1) * y_split_length ] ]

    x_folds += [ x_data[ (k - 1) * x_split_length : len(x_data) ] ] 
    y_folds += [ y_data[ (k - 1) * y_split_length : len(y_data) ] ]

    for fold_index in range(k):
        
        x_train = []
        y_train = []

        for train_index in range(k):
            if train_index != fold_index:
                x_train.extend(x_folds[train_index])
                y_train.extend(y_folds[train_index])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_test = x_folds[fold_index]
        y_test = y_folds[fold_index]
    
        yield x_train, x_test, y_train, y_test

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