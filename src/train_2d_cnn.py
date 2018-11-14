from model import ECGModel2D

from utils import prepare_image_data, k_fold, read_csv, plot_loss

from settings import PREDICTION_LABELS, EPOCHS, PLOT_FILE
from settings import GROUND_TRUTH_PATH, MEDIANS_IMAGE_PATH
from settings import X_TRANSPOSE, BATCH_SIZE, SEED, K_FOLDS
from settings import EXPERIMENT_NAME, EXPERIMENT_ROOT, MODEL_FILE

import os

try:
    from sacred import Experiment
    from sacred.utils import apply_backspaces_and_linefeeds
    from sacred.observers import FileStorageObserver
except:
    Experiment = None  

INPUT_SHAPE = (8, 600, 3)

def train():

    data = read_csv(
        csv_file=GROUND_TRUTH_PATH,
        delimiter=';',
        transpose=False,
        skip_header=True)

    x_data, y_data = prepare_image_data(
        data=data,
        prediction_labels=PREDICTION_LABELS,
        training_path=MEDIANS_IMAGE_PATH, 
        x_shape=INPUT_SHAPE)

    for fold_index, (x_train, x_test, y_train, y_test) in enumerate(k_fold(x_data, y_data, K_FOLDS)):

        model = ECGModel2D(input_shape=INPUT_SHAPE, output_size=len(PREDICTION_LABELS))

        model.compile(
            loss='mse',
            optimizer='nadam',
            metrics=[ 'mse' ])

        history = model.fit(x_train, y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=1,
                validation_data=(x_test, y_test),
                shuffle=True)
        
        model.save(f'model_{ fold_index }.h5')

        plot_loss(history, PLOT_FILE)

        model_path = f'{ os.path.splitext(MODEL_FILE)[0] }_{ fold_index }.h5'
        model.save(model_path)

        experiment.add_artifact(PLOT_FILE)
        experiment.add_artifact(model_path)
        
        score = model.evaluate(x_test, y_test, verbose=0)


if __name__ == '__main__':

    if Experiment is not None:

        experiment_path = f'{EXPERIMENT_ROOT}/{ EXPERIMENT_NAME }'

        experiment = Experiment(EXPERIMENT_NAME)
        experiment.captured_out_filter = apply_backspaces_and_linefeeds
        experiment.observers.append(FileStorageObserver.create(experiment_path))
        
        experiment.automain( train )

    else:
        train()