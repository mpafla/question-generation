import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, questions, answers, sources, max_sequence_length, n_classes, n_samples, batch_size=32,
                 shuffle=True):
        'Initialization'
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.questions = questions
        self.answers = answers
        self.sources = sources
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.shuffle = shuffle
        self.indexes = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def generateTestBatch(self):
        return self.__data_generation([0,1,2])

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, Y = self.__data_generation(indexes)

        return X, Y

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim)
        # Initialization
        questions_batch = np.empty((self.batch_size, self.max_sequence_length, self.n_classes))
        answers_batch = np.empty((self.batch_size, self.max_sequence_length))
        sources_batch = np.empty((self.batch_size, self.max_sequence_length))

        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            questions_batch[i] = keras.utils.to_categorical(self.questions[index], num_classes=self.n_classes)
            answers_batch[i] = self.answers[index]
            sources_batch[i] = self.sources[index]

        #return [answers_batch, sources_batch], questions_batch
        return answers_batch, questions_batch