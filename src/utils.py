import numpy as np

def z_normalize(data):
    (np.array(data) - np.mean(data)) / np.std(data)
