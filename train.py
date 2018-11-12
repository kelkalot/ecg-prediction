import csv
import numpy as np

DATA_LABELS = {
    'TestID': 0,
    'VentricularRate': 1,
    'P_RInterval': 2,
    'QRSDuration': 3,
    'Q_TInterval': 4,
    'QTCCalculation': 5,
    'PAxis': 6,
    'RAxis': 7,
    'TAxis': 8,
    'QRSCount': 9,
    'QOnset': 10,
    'QOffset': 11,
    'POnset': 12,
    'POffset': 13,
    'TOffset': 14
}

def read_csv(csv_file, delimiter=',', skip_header=True, as_type='np_array'):

    data = []

    with open(csv_file, 'r') as f:
        csv_data = csv.reader(f, delimiter=delimiter)
        
        if skip_header:
            next(csv_data)
        
        for row in csv_data:
            data.append(row)

    if as_type == 'np_array':
        data = np.array(data)

    return data

def z_normalize(data):
    (np.array(data) - np.mean(data)) / np.std(data)

def prepare_data(data, label='QOnset'):

    index = DATA_LABELS[label]

    x_data = []
    y_data = []

    for row in data:
        try:
            median_data = read_csv(f'./medians/{ row[0] }.asc', ' ')
            y_data.append(row[index])
            x_data.append(median_data)
        except FileNotFoundError:
            pass
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

def split_data(data, ratio=.3):
    split_index = int(len(data) - (len(data) * ratio))
    return data[:split_index], data[split_index:]

def build_model(input_shape=(600, 8)):
    model = Sequential()
    model.add(Convolution1D(input_shape = input_shape,
                            nb_filter=16,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Convolution1D(nb_filter=8,
                            filter_length=4,
                            border_mode='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dense(1))

    return model

if __name__ == '__main__':

    data = read_csv('./ground_truth.csv')

    x_data, y_data = prepare_data(data)

    x_train, x_test = split_data(x_data)
    y_train, y_test = split_data(y_data)

    model = build_model()
    opt = Nadam(lr=0.002)

    model.compile(optimizer=opt, 
              loss='mean_squared_error',
              metrics=['accuracy'])

    history = model.fit(x_train, y_train, 
          nb_epoch = 100, 
          batch_size = 128, 
          verbose=1, 
          validation_data=(y_test, y_test),
          shuffle=True)

    read_csv('./ground_truth.csv')
    read_csv('./medians/101.asc', ' ')



    
    