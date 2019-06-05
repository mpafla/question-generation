import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from trainers.trainer import Trainer
from trainers.learning_schudule import CustomSchedule
from tensorflow.python.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt

class TrainerTransformer(Trainer):
    def __init__(self, config, vocab):
        super(TrainerTransformer, self).__init__(config, vocab)
        self.d_model = config["model"]["d_model"]
        self.learning_rate_schedule = CustomSchedule(self.d_model)
        self.beta_1 = config["trainer"]["beta_1"]
        self.beta_2 = config["trainer"]["beta_2"]
        self.epsilon = config["trainer"]["epsilon"]
        self.optimizer = getattr(tf.optimizers, config["trainer"]["optimizer"])(self.learning_rate_schedule, self.beta_1, self.beta_2, 1e-9)
        self.loss_object = SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    @tf.function
    def train_step(self, input, target, model, training):
        input_answer = input[:, 0]
        input_context = input[:, 1]
        target_input = target[:, :-1]
        target_real = target[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(input_context, target_input)

        with tf.GradientTape() as tape:
            predictions, _ = model(input_context, target_input,
                                         training,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)


            #real_string = [self.vocab.get_token_for_index(index.numpy()) for index in target_real[0]]
            #print(real_string)

            #pred = np.argmax(predictions, axis=2)
            #pred_string = [self.vocab.get_token_for_index(index) for index in pred[0]]
            #print(pred_string)

            loss = self.loss_function(target_real, predictions)

        if training:
            gradients = tape.gradient(loss, model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            self.train_loss(loss)
            self.train_accuracy(target_real, predictions)
        else:
            self.test_loss(loss)
            self.test_accuracy(target_real, predictions)
        

    def loss_function(self, real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    def train(self, model, dataset_train, dataset_dev):
        dataset_train = dataset_train.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        dataset_dev = dataset_dev.batch(self.batch_size)


        print("Start training")

        print("Pre-loading parts of dataset")
        for epoch in range(self.epochs):
            start = time.time()

            for (batch, (input, target)) in enumerate(dataset_train):

                self.train_step(input, target, model, training=True)

                if batch % 100 == 0:
                    print(self.diagnostics(epoch, batch))
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()


            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for (batch, (input, target)) in enumerate(dataset_dev):
                self.train_step(input, target, model, training=False)

            print(self.diagnostics(epoch))

            #if (epoch + 1) % 5 == 0:
            #    ckpt_save_path = ckpt_manager.save()
            #   print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
            #                                                        ckpt_save_path))

            print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask


    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

    #masks all the future tokens because they are irrelevant for current token - creates a triangular matrix because for the first word all other words are masked, for the second only the first word is not masked, and so on
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)