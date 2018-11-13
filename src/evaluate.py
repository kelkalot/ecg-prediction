from keras.models import load_model
from train import read_csv
import numpy as np

model = load_model('../model.h5')

data = []
data.append(read_csv('/Users/stevenah/github/ecg-prediction/data/medians/101.asc', ' '))
data = np.array(data)

print(model.predict(data))