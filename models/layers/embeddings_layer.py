from tensorflow.python.keras.layers import Embedding

def get_embeddings_layer(embeddings_matrix):
    vocab_size = embeddings_matrix.shape[0]
    embeddings_dim = embeddings_matrix.shape[1]
    embedding_layer = Embedding(vocab_size,
                      embeddings_dim,
                      mask_zero=False,
                      weights=[embeddings_matrix],
                      trainable=False)
    return embedding_layer