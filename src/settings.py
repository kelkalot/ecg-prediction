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
BATCH_SIZE = 16
SEED = 2
K_FOLDS = 3

X_TRANSPOSE = True

GROUND_TRUTH_PATH = '/home/steven/github/ecg-prediction/data/ground_truth.csv'
MEDIANS_PATH = '/home/steven/github/ecg-prediction/data/medians'
RHYTHM_PATH = '/home/steven/github/ecg-prediction/data/rhythm'

MODEL_FILE = '/home/steven/github/ecg-prediction/model.h5'

LOSS_FUNCTION = 'mse'
OPTIMIZER = Nadam(lr=0.001)

PLOT_FILE = 'loss_plot.png'