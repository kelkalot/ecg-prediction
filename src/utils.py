import numpy as np

try:
    import matplotlib.pyplot as plt
except:
    plt = None

def z_normalize(data):
    (np.array(data) - np.mean(data)) / np.std(data)


def plot_loss(history, file_name):

    if plt is None:
        return
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.legend(['training', 'validation'], loc='upper left')

    plt.savefig(file_name)
    plt.gcf().clear()
