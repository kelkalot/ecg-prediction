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

EPOCHS = 1
BATCH_SIZE = 16
SEED = 2

X_TRANSPOSE = False

GROUND_TRUTH_PATH = '/Users/stevenah/github/ecg-prediction/data/ground_truth.csv'
MEDIANS_PATH = '/Users/stevenah/github/ecg-prediction/data/medians'
RHYTHM_PATH = '/Users/stevenah/github/ecg-prediction/data/rhythm'

MODEL_FILE = '/Users/stevenah/github/ecg-prediction/model.h5'

LOSS_FUNCTION = 'mse'
OPTIMIZER = 'nadam'

PLOT_FILE = 'loss_plot.png'