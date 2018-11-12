from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

from keras.models import Model

class ECGModel():

    def __init__(self, input_shape=(599, 8), output_size=3):
        self.input_shape = input_shape
        self.output_size = output_size

        self.build_model()

    def fit(self, x_train, y_train, **kwargs):
        return self.model.fit(x_train, y_train, **kwargs)

    def compile(self, **kwargs):
        self.model.compile(**kwargs)

    def save(self, path):
        self.model.save(path)

    def build_model(self):
        inputs = Input(shape=self.input_shape)

        x = Dense(64, activation='relu')(inputs)

        x = Conv1D(100, 10, activation='relu')(x)
        x = Conv1D(100, 10, activation='relu')(x)

        x = MaxPooling1D(3)(x)

        x = Conv1D(160, 10, activation='relu')(x)
        x = Conv1D(160, 10, activation='relu')(x)

        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.5)(x)
        
        output = Dense(self.output_size)(x)

        self.model = Model(inputs=inputs, outputs=output)

    # def build_model(self):
    #     inputs = Input(shape=self.input_shape)

    #     x = Dense(64, activation='relu')(inputs)
    #     x = Dense(64, activation='relu')(inputs)

    #     x = Flatten()(x)
    #     output = Dense(self.output_size)(x)

    #     self.model = Model(inputs=inputs, outputs=output)
