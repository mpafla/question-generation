from tensorflow.python.keras.layers import Layer, Dense

class PointWiseFeedForwardNetwork(Layer):
    def __init__(self, d_model, dff):
        super(PointWiseFeedForwardNetwork, self).__init__()

        self.d1 = Dense(dff, activation='relu')
        self.d2 = Dense(d_model)

    def call(self, inputs):
        d1_output = self.d1(inputs)
        d2_output = self.d2(d1_output)
        return d2_output