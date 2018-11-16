from keras.optimizers import Nadam

from model import ECGModel1D

from utils import plot_loss, read_csv
from utils import split_data, prepare_csv_data, k_fold

try:
    from sacred import Experiment
    from sacred.utils import apply_backspaces_and_linefeeds
    from sacred.observers import FileStorageObserver
except:
    Experiment = None    

import numpy as np
import tensorflow as tf
import keras.backend as K
import os

from settings import PREDICTION_LABELS, EPOCHS, PLOT_FILE
from settings import X_TRANSPOSE, BATCH_SIZE, SEED, K_FOLDS
from settings import GROUND_TRUTH_PATH, MEDIANS_PATH, RHYTHM_PATH
from settings import MODEL_FILE, LOSS_FUNCTION, OPTIMIZER, METRICS
from settings import EXPERIMENT_NAME, EXPERIMENT_ROOT

tf.set_random_seed(SEED)
np.random.seed(SEED)

def train():
    data = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    x_data, y_data = prepare_csv_data(
        data=data,
        prediction_labels=PREDICTION_LABELS,
        training_path=MEDIANS_PATH, 
        x_shape=X_TRANSPOSE)

    for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, K_FOLDS)):
    
        model = ECGModel1D(
            input_shape=x_data[0].shape,
            output_size=len(PREDICTION_LABELS))

        model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS_FUNCTION,
            metrics=METRICS)
            
        history = model.fit(x_train, y_train, 
            epochs=EPOCHS,
            batch_size= BATCH_SIZE,
            verbose=1,
            validation_data=(x_test, y_test),
            shuffle=True)

        plot_loss(history, PLOT_FILE)

        model_path = f'{ os.path.splitext(MODEL_FILE)[0] }_{ fold_index }.h5'
        model.save(model_path)

        if Experiment is not None:
            experiment.add_artifact(PLOT_FILE)
            experiment.add_artifact(model_path)

if __name__ == '__main__':

    if Experiment is not None:

        experiment_path = f'{EXPERIMENT_ROOT}/{ EXPERIMENT_NAME }'

        experiment = Experiment(EXPERIMENT_NAME)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))
        
        experiment.automain( train )

    else:
        train()


    
    