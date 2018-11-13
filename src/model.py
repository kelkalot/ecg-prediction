from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D

from keras.models import Model

class ECGModel():

    def __init__(self, input_shape=(8, 599), output_size=3):
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

        x = Conv1D(1200, 2, activation='relu')(inputs)
        x = Conv1D(600, 2, activation='relu')(x)

        x = MaxPooling1D(2)(x)

        x = Conv1D(200, 2, activation='relu')(x)
        x = Conv1D(200, 2, activation='relu')(x)
       # x = Conv1D(200, 3, activation='relu')(x)

        x = GlobalAveragePooling1D()(x)
#        x = Dropout(0.5)(x)
        
        output = Dense(self.output_size)(x)

        model = Model(inputs=inputs, outputs=output)

        print(model.summary())

        self.model = model