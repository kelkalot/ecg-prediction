
import numpy as np
import os

from utils import read_csv
from PIL import Image

ECG_DATA_ROOT = '/home/steven/Dropbox/Github/ecg-prediction/data/medians'

IMAGE_WIDTH = 600
IMAGE_HEIGHT = 8
IMAGE_CHANNELS = 3
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

R_CHANNEL = 0
G_CHANNEL = 1
B_CHANNEL = 2

def generate_ecg_image(ecg_data, ecg_id, save_path='../data/medians_images'):
        
    canvas = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    
    for y_index in range(canvas.shape[0]):
        for x_index in range(canvas.shape[1]):

            # RED channel
            if ecg_data[ y_index, x_index ] < 0:
                canvas[ y_index, x_index, R_CHANNEL ] = min(255, ecg_data[ y_index, x_index ] * -1)

            # GREEN channel
            if ecg_data[ y_index, x_index ] > 0:
                canvas[ y_index, x_index, G_CHANNEL ] = min(255, ecg_data[ y_index, x_index ])

            # BLUE channel
            if ecg_data[ y_index, x_index ] == 0:
                canvas[ y_index, x_index, B_CHANNEL ] = 255

        # Create the image
        image = Image.fromarray(canvas, 'RGB')
        image.save(f'{ save_path }/{ ecg_id }.png')

if __name__ == '__main__':

    for ecg_data in os.listdir(ECG_DATA_ROOT):
        ecg = read_csv(os.path.join(ECG_DATA_ROOT, ecg_data), delimiter=' ', transpose=True, skip_header=False)
        generate_ecg_image(ecg, os.path.splitext(ecg_data)[0])
    
