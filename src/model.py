from keras.layers import Input, Dense, Flatten
from keras.models import Sequential
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

        x = Dense(32, activation='relu')(inputs)
        x = Dense(64, activation='relu')(inputs)
        x = Dense(128, activation='relu')(inputs)

        x = Flatten()(x)
        output = Dense(self.output_size)(x)

        self.model = Model(inputs=inputs, outputs=output)