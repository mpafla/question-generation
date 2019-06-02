import tensorflow as tf
from models.base_model import BaseModel
from models.layers.bahdanau_attention_layer import BahdanauAttentionLayer
#from models.layers.embeddings_layer import get_embeddings_layer
from tensorflow.python.keras.layers import Input, CuDNNGRU, Add, Dense, Embedding
from tensorflow.python.keras.models import Model

class GRU_Bahdanau(BaseModel):
    def __init__(self, config, embeddings_matrix):
        self.gru_hidden_states = config["model"]["gru_hidden_states"]
        self.attention_layer_units = config["model"]["attention_layer_units"]
        self.vocabulary_size = embeddings_matrix.shape[0]

        super(GRU_Bahdanau, self).__init__(config, embeddings_matrix)

    def get_embeddings_layer(self, embeddings_matrix):
        vocab_size = embeddings_matrix.shape[0]
        embeddings_dim = embeddings_matrix.shape[1]
        embedding_layer = Embedding(vocab_size,
                                    embeddings_dim,
                                    mask_zero=False,
                                    weights=[embeddings_matrix],
                                    trainable=False)
        return embedding_layer

    def set_up_model(self):
        encoder_input_answers = Input(shape=(None,))
        encoder_input_sources = Input(shape=(None,))
        encoder_embedding_answers = self.get_embeddings_layer(self.embeddings_matrix)(encoder_input_answers)
        encoder_embedding_source = self.get_embeddings_layer(self.embeddings_matrix)(encoder_input_sources)
        encoder_input = Add()([encoder_embedding_answers, encoder_embedding_source])
        encoder_gru = CuDNNGRU(self.gru_hidden_states, return_sequences=True, return_state=True)
        encoder_output, state_h_encoder = encoder_gru(encoder_input)

        # encoder_states = [state_h, state_c]

        decoder_input_questions = Input(shape=(None,))
        decoder_embedding_questions = self.get_embeddings_layer(self.embeddings_matrix)(decoder_input_questions)

        decoder_hidden_states_input = Input(shape=(self.gru_hidden_states,))
        # print(decoder_hidden_states_input.shape)
        decoder_encoder_output_input = Input(shape=(None, encoder_output.shape[2]))
        # print(decoder_encoder_output_input.shape)
        attention_layer = BahdanauAttentionLayer(self.attention_layer_units)
        context_vector, attention_weights = attention_layer((decoder_hidden_states_input, decoder_encoder_output_input))

        # context_vector = tf.reshape(context_vector,
        #                           [tf.shape(decoder_embedding_questions)[0], tf.shape(decoder_embedding_questions)[1], context_vector.shape[1]])
        # print(context_vector.shape )
        # print(decoder_embedding_questions.shape)
        decoder_inputs = tf.concat([tf.expand_dims(context_vector, 1), decoder_embedding_questions], axis=-1)
        # print(decoder_inputs.shape)
        # decoder_input_attention = tf.concat([context_vector, decoder_embedding_questions],axis=-1)

        decoder_gru = CuDNNGRU(self.gru_hidden_states, return_sequences=True, return_state=True)
        decoder__gru_outputs, decoder_gru_hidden = decoder_gru(decoder_inputs)
        decoder__gru_outputs = tf.reshape(decoder__gru_outputs, (-1, decoder__gru_outputs.shape[2]))
        # decoder_outputs_flattend = Flatten()(decoder_outputs)
        decoder_dense = Dense(self.vocabulary_size, activation='softmax')
        decoder_outputs = decoder_dense(decoder__gru_outputs)

        # model = Model([encoder_input_answers, encoder_input_sources, decoder_input_questions], decoder_outputs)
        # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        # model.summary()

        #encoder = Model([encoder_input_answers, encoder_input_sources], [encoder_output, state_h_encoder])
        encoder = Model([encoder_input_answers, encoder_input_sources], [encoder_embedding_answers, encoder_embedding_source])
        encoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        encoder.summary()

        decoder = Model([decoder_input_questions, decoder_hidden_states_input, decoder_encoder_output_input],
                        [decoder_outputs, decoder_gru_hidden])
        decoder.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        decoder.summary()

        return encoder, decoder