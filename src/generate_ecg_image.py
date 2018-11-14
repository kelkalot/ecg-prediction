
import numpy as np
import os

from utils import read_csv
from PIL import Image

from settings import MEDIANS_PATH, IMAGE_SHAPE

R_CHANNEL = 0
G_CHANNEL = 1
B_CHANNEL = 2

BASELINE_UPPER_LIMIT = 50
BASELINE_LOWER_LIMIT = -20
BLUE_STRENGTH = 0

def generate_ecg_image(ecg_data, ecg_id, save_path='../data/medians_images'):
        
    canvas = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for y_index in range(canvas.shape[0]):
        for x_index in range(canvas.shape[1]):

            ecg_value = ecg_data[ y_index, x_index ]

            # RED channel
            if ecg_value < BASELINE_LOWER_LIMIT:
                canvas[ y_index, x_index, R_CHANNEL ] = min(255, abs(ecg_value))

            # GREEN channel
            if ecg_value > BASELINE_UPPER_LIMIT:
                canvas[ y_index, x_index, G_CHANNEL ] = min(255, ecg_value)

            # BLUE channel
            if ecg_value <= BASELINE_UPPER_LIMIT and ecg_value >= BASELINE_LOWER_LIMIT:
                canvas[ y_index, x_index, B_CHANNEL ] =  abs(BLUE_STRENGTH - abs(ecg_value))

    # Create the image
    image = Image.fromarray(canvas, 'RGB')
    image.save(f'{ save_path }/{ ecg_id }.png')

if __name__ == '__main__':

    for index, ecg_file in enumerate(os.listdir(MEDIANS_PATH)):
        print(f'Generating images: { index + 1 } / { len(os.listdir(MEDIANS_PATH)) }', end='\r')
        ecg_data = read_csv(os.path.join(MEDIANS_PATH, ecg_file), delimiter=' ', transpose=True, skip_header=False, dtype=np.int)
        generate_ecg_image(ecg_data, os.path.splitext(ecg_file)[0])
    
