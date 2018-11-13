from keras.models import load_model
from train import read_csv
from utils import k_fold, prepare_data, read_csv

import keras.backend as K
import numpy as np
import os

from settings import PREDICTION_LABELS, EPOCHS, PLOT_FILE
from settings import X_TRANSPOSE, BATCH_SIZE, SEED, K_FOLDS
from settings import GROUND_TRUTH_PATH, MEDIANS_PATH, RHYTHM_PATH
from settings import MODEL_FILE, LOSS_FUNCTION, OPTIMIZER

GROUND_TRUTH = read_csv(
    csv_file=GROUND_TRUTH_PATH,
    delimiter=';',
    transpose=False,
    skip_header=True)

x_data, y_data = prepare_data(
    data=GROUND_TRUTH,
    prediction_labels=PREDICTION_LABELS,
    training_path=MEDIANS_PATH, 
    x_shape=X_TRANSPOSE)


for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, K_FOLDS)):
    
    model = load_model(f'{ os.path.splitext(MODEL_FILE)[0] }_{ fold_index }.h5')
    
    for row_index, row in enumerate(x_test):
        prediction = np.round(model.predict(row.reshape((1, *row.shape))).flatten())
        
        print(prediction, y_test[row_index])