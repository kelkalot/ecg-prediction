
import numpy as np
import os
import shutil

from utils import read_csv, normalize
from PIL import Image

from settings import MEDIANS_PATH, IMAGE_SHAPE, MEDIANS_IMAGE_PATH

R_CHANNEL = 0
G_CHANNEL = 1
B_CHANNEL = 2

BASELINE_UPPER_LIMIT = 50
BASELINE_LOWER_LIMIT = -20
BLUE_STRENGTH = 0

Y_MAX = 255

def generate_ecg_image(ecg_data, ecg_id, save_path):
        
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

    if os.path.exists(MEDIANS_IMAGE_PATH):
        shutil.rmtree(MEDIANS_IMAGE_PATH)

    for index, ecg_file in enumerate(os.listdir(MEDIANS_PATH)):
        print(f'Generating images: { index + 1 } / { len(os.listdir(MEDIANS_PATH)) }', end='\r')
        
        # Read ECG data from .asc file
        ecg_data = read_csv(
            os.path.join(MEDIANS_PATH, ecg_file),
            delimiter=' ',
            transpose=True,
            skip_header=False,
            dtype=np.int)

        # Normalize ECG data between -1 and 1
        ecg_normalized = normalize(ecg_data)

        # Scale normlaized ECG data between -Y_MAX and Y_MAX
        ecg_scaled = np.array(ecg_normalized * Y_MAX, dtype=int)

        # Generate ECG image
        generate_ecg_image(ecg_data, os.path.splitext(ecg_file)[0], MEDIANS_IMAGE_PATH)
    
    print('\n', end='\r')