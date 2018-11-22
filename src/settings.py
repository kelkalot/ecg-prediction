from keras.optimizers import Nadam
from keras.callbacks import TensorBoard

import os

DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCCalculation': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14
}


ROOT_DIRECTORY = '/home/steven/Dropbox/Github/ecg-prediction'

PREDICTION_LABELS = [
    'TOffset'
]

EPOCHS = 5000
BATCH_SIZE = 16
SEED = 2
K_FOLDS = 3

X_TRANSPOSE = True
IMAGE_SHAPE = (8, 600, 3)

METRICS = [ 'mse' ]

MODEL_CALLBACKS = [
    TensorBoard(log_dir='../logs', batch_size=BATCH_SIZE)
]

PLOT_FILE = 'loss_plot.png'
MODEL_FILE = 'model.h5'

GROUND_TRUTH_PATH = os.path.join(ROOT_DIRECTORY, 'data/ground_truth.csv')
MEDIANS_PATH = os.path.join(ROOT_DIRECTORY, 'data/medians')
RHYTHM_PATH = os.path.join(ROOT_DIRECTORY, 'data/rhythm')
MEDIANS_IMAGE_PATH = os.path.join(ROOT_DIRECTORY, 'data/medians_images')
MEDIANS_FEATURE_PATH = os.path.join(ROOT_DIRECTORY, 'data/medians_features')
MEDIANS_PLOT_PATH = os.path.join(ROOT_DIRECTORY, 'data/medians_plots')
EXPERIMENT_ROOT = os.path.join(ROOT_DIRECTORY, 'experiments')
MODEL_PATH = os.path.join(ROOT_DIRECTORY, MODEL_FILE)

EXPERIMENT_NAME = 'ECG_CNN_MODEL'

LOSS_FUNCTION = 'mean_squared_logarithmic_error'
OPTIMIZER = 'nadam'


