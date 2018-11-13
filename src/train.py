from keras.optimizers import Nadam

from model import ECGModel

from utils import plot_loss, read_csv
from utils import split_data, prepare_data

import numpy as np
import tensorflow as tf
import keras.backend as K

PREDICTION_LABELS = [
    'QOnset',
    'QOffset',
    'POnset',
    'POffset',
    'TOffset'
]

EPOCHS = 100
BATCH_SIZE = 16
SEED = 2

X_SHAPE = (600, 8)

GROUND_TRUTH_PATH = '/Users/stevenah/github/ecg-prediction/data/ground_truth.csv'
MEDIANS_PATH = '/Users/stevenah/github/ecg-prediction/data/medians'
RHYTHM_PATH = '/Users/stevenah/github/ecg-prediction/data/rhythm'
MODEL_FILE = '/Users/stevenah/github/ecg-prediction/model.h5'

LOSS_FUNCTION = 'mse'
PLOT_FILE = 'loss_plot.png'

tf.set_random_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':

    data = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        shape=None,
        skip_header=True)

    x_data, y_data = prepare_data(
        data=data,
        prediction_labels=PREDICTION_LABELS,
        training_path=MEDIANS_PATH, 
        x_shape=X_SHAPE)

    x_train, x_test = split_data(x_data)
    y_train, y_test = split_data(y_data)

    model = ECGModel(
        input_shape=x_data[0].shape,
        output_size=len(PREDICTION_LABELS))

    optimizer = Nadam(lr=0.001)

    model.compile(optimizer=optimizer, loss=LOSS_FUNCTION)

    history = model.fit(x_train, y_train, 
          epochs=EPOCHS,
          batch_size= BATCH_SIZE,
          verbose=1,
          validation_data=(x_test, y_test),
          shuffle=True)

    plot_loss(history, PLOT_FILE)

    model.save(MODEL_FILE)



    
    