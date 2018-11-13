from keras.models import load_model
from train import read_csv

import keras.backend as K
import numpy as np




model = load_model('../model.h5')

data = []
data.append(read_csv('/Users/stevenah/github/ecg-prediction/data/medians/101.asc', ' ', False, False))
data = np.array(data)

print(np.round(model.predict(data).flatten()))