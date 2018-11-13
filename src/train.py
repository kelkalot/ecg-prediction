from keras.optimizers import Nadam

from model import ECGModel, ECGBetterModel

from utils import plot_loss, read_csv
from utils import split_data, prepare_data, k_fold

import numpy as np
import tensorflow as tf
import keras.backend as K
import os

from settings import PREDICTION_LABELS, EPOCHS, PLOT_FILE
from settings import X_TRANSPOSE, BATCH_SIZE, SEED, K_FOLDS
from settings import GROUND_TRUTH_PATH, MEDIANS_PATH, RHYTHM_PATH
from settings import MODEL_FILE, LOSS_FUNCTION, OPTIMIZER

tf.set_random_seed(SEED)
np.random.seed(SEED)

if __name__ == '__main__':

    data = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    x_data, y_data = prepare_data(
        data=data,
        prediction_labels=PREDICTION_LABELS,
        training_path=MEDIANS_PATH, 
        x_shape=X_TRANSPOSE)

    fold_index = 0

    for x_train, x_test, y_train, y_test in k_fold(x_data, y_data, K_FOLDS):
    
        model = ECGBetterModel(
            input_shape=x_data[0].shape,
            output_size=len(PREDICTION_LABELS))

        model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS_FUNCTION)
            
        history = model.fit(x_train, y_train, 
                epochs=EPOCHS,
                batch_size= BATCH_SIZE,
                verbose=1,
                validation_data=(x_test, y_test),
                shuffle=True)

        plot_loss(history, PLOT_FILE)

        model.save(f'{ os.path.splitext(MODEL_FILE)[0] }_{ fold_index }.h5')

        fold_index += 1



    
    