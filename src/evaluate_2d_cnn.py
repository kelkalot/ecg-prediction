from keras.models import load_model
from utils import k_fold, prepare_image_data, read_csv

import keras.backend as K
import numpy as np
import os

from settings import PREDICTION_LABELS, IMAGE_SHAPE, K_FOLDS
from settings import GROUND_TRUTH_PATH, MEDIANS_IMAGE_PATH, MODEL_PATH

GROUND_TRUTH = read_csv(
    csv_file=GROUND_TRUTH_PATH,
    delimiter=';',
    transpose=False,
    skip_header=True)

x_data, y_data = prepare_image_data(
    data=GROUND_TRUTH,
    prediction_labels=PREDICTION_LABELS,
    training_path=MEDIANS_IMAGE_PATH, 
    x_shape=IMAGE_SHAPE)


for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, K_FOLDS)):
    
    model = load_model(f'{ os.path.splitext(MODEL_PATH)[0] }_{ fold_index }.h5')
    
    for row_index, row in enumerate(x_test):
        prediction = np.round(model.predict(row.reshape((1, *row.shape))).flatten())
        
        print(prediction, y_test[row_index])