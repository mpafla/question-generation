
class BaseModel():
    # To build your model, you only to pass a "configuration" which is a dictionary
    def __init__(self, config, embeddings_matrix):
        self.config = config
        self.embeddings_matrix = embeddings_matrix
        self.random_seed = self.config["model"]['random_seed']
        self.encoder, self.decoder = self.set_up_model()
        # At the end of this function, you want your model to be ready!

    def set_up_model(self):
        raise Exception('The model needs to be set up')

    def get_encoder_and_decoder(self):
        return self.encoder, self.decoder