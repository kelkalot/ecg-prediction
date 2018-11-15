from keras.optimizers import Nadam


DATA_LABELS = {
    'TestID': 0, 'VentricularRate': 1, 'P_RInterval': 2, 'QRSDuration': 3,
    'Q_TInterval': 4, 'QTCCalculation': 5, 'PAxis': 6, 'RAxis': 7, 'TAxis': 8,
    'QRSCount': 9, 'QOnset': 10, 'QOffset': 11, 'POnset': 12, 'POffset': 13,
    'TOffset': 14
}

PREDICTION_LABELS = [
    'QOnset',
    'QOffset',
    'POnset',
    'POffset',
    'TOffset'
]

EPOCHS = 1000
BATCH_SIZE = 1
SEED = 2
K_FOLDS = 3

X_TRANSPOSE = True
IMAGE_SHAPE = (8, 600, 3)

METRICS = [ 'mse' ]

GROUND_TRUTH_PATH = '/Users/stevenah/github/ecg-prediction/data/ground_truth.csv'
MEDIANS_PATH = '/Users/stevenah/github/ecg-prediction/data/medians'
RHYTHM_PATH = '/Users/stevenah/github/ecg-prediction/data/rhythm'

MEDIANS_IMAGE_PATH = '/Users/stevenah/github/ecg-prediction/data/medians_images'
MEDIANS_FEATURE_PATH = '/Users/stevenah/github/ecg-prediction/data/medians_features'

EXPERIMENT_NAME = 'ECG_CNN_MODEL'
EXPERIMENT_ROOT = '/Users/stevenah/github/ecg-prediction/experiments'

MODEL_FILE = '/Users/stevenah/github/ecg-prediction/model.h5'

LOSS_FUNCTION = 'mean_squared_logarithmic_error'
OPTIMIZER = Nadam(lr=0.001)

PLOT_FILE = 'loss_plot.png'

