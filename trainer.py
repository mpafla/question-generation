
import tensorflow as tf
import os
import numpy as np
from utils.constants import Constants
import time



class Trainer():
    def __init__(self, config, vocab):
        self.vocab = vocab
        self.epochs = config["trainer"]["epochs"]
        self.save_after = config["trainer"]["save_after"]
        self.batch_size = config["trainer"]["batch_size"]
        self.shuffle_buffer_size = config["trainer"]["shuffle_buffer_size"]
        self.learning_rate = config["trainer"]["learning_rate"]
        self.loss_function = getattr(tf.keras.losses, config["trainer"]["loss_function"])()
        self.train_accuracy = getattr(tf.keras.metrics, config["trainer"]["accuracy"])()
        self.test_accuracy = getattr(tf.keras.metrics, config["trainer"]["accuracy"])()
        self.train_loss = getattr(tf.keras.metrics, config["trainer"]["loss_metric"])()
        self.test_loss = getattr(tf.keras.metrics, config["trainer"]["loss_metric"])()
        self.optimizer = getattr(tf.optimizers, config["trainer"]["optimizer"])(lr=self.learning_rate)
        self.checkpoint_dir = 'models/saved_models/' + config["model"]["file_name"]
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.training = config["trainer"]["training"]

    def diagnostics(self, epoch, step):
        template = 'Epoch: {}/{}, Step: {}/{}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        diagnostics = template.format(epoch + 1,
                                      self.epochs,
                                      step,
                                      int(98000 / self.batch_size),
                                      self.train_loss.result(),
                                      self.train_accuracy.result() * 100,
                                      self.test_loss.result(),
                                      self.test_accuracy.result() * 100)
        return diagnostics

    @tf.function
    def run_through_step(self, encoder, decoder, input, target_one_hot, training=True):
        start = time.time()
        loss = 0
        answers = input[:,0]
        contexts = input[:, 1]

        targets = tf.math.argmax(target_one_hot, axis=2)

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder.call([answers, contexts])

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.vocab.get_index_for_token(Constants.SOS)] * target_one_hot.shape[0], 1)
            # dec_input = to_categorical([special_token2index(vocabulary_size - 4, SOS)] * BATCH_SIZE)

            #target_predictions = tf.Variable(np.zeros(target_one_hot.shape), dtype=tf.float32)
            target_predictions = None

            # Teacher forcing - feeding the target as the next input
            for t in range(target_one_hot.shape[1]):
                # passing enc_output to the decoder

                predictions, dec_hidden = decoder.call([dec_input, dec_hidden, enc_output])

                predictions_exp = tf.expand_dims(predictions, 1)


                if target_predictions is None:
                    target_predictions = predictions_exp
                else:
                    target_predictions = tf.concat([target_predictions, predictions_exp], axis= 1)


                #print(predictions)
                #print(target_predictions)
                #print(target_predictions[:, t])

                #target_predictions[:, t].assign(predictions)

                loss += self.loss_function(target_one_hot[:, t], predictions)

                # if training:
                #    train_accuracy(target_one_hot[:, t], predictions)
                # else:
                #    test_accuracy(target_one_hot[:, t], predictions)

                # using teacher forcing
                dec_input = tf.expand_dims(targets[:, t], 1)

        batch_loss = (loss / int(targets.shape[1]))


        if training:
            try:
                variables = encoder.trainable_variables + decoder.trainable_variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))
                self.train_accuracy(target_one_hot, target_predictions)
                self.train_loss(batch_loss)
            except:
                print("Optimazation could not be performed")
        else:
            self.test_loss(batch_loss)
            self.test_accuracy(target_one_hot, target_predictions)

        return (target_predictions)



    def train(self, model, dataset_train, dataset_test):
        dataset_train = dataset_train.shuffle(self.shuffle_buffer_size).batch(self.batch_size).prefetch(self.batch_size)
        dataset_test = dataset_test.batch(self.batch_size)

        encoder, decoder = model.get_encoder_and_decoder()

        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                         encoder=encoder,
                                         decoder=decoder)
        checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        print("Start training")
        for epoch in range(self.epochs):
            step = 0
            start = time.time()

            print("Training the model")
            for data in dataset_train:
                input = data[0]
                target_one_hot = data[1]

                self.run_through_step(encoder, decoder, input, target_one_hot, training=True)
                step = step + 1

                # if (step % 10 == 0):
                #    print(diagnostics(epoch, step))
                print(self.diagnostics(epoch, step))

            print("Getting accuracy from test dataset")
            for data in dataset_test:
                input = data[0]
                target_one_hot = data[1]

                self.run_through_step(encoder, decoder, input, target_one_hot, training=False)

            print(self.diagnostics(epoch, step))
            print('Time taken for training this epoch is {} sec'.format(time.time() - start))

            with open("models/diagnostics.txt", "a") as text_file:
                print(self.diagnostics(epoch, step), file=text_file)

            #if (epoch % self.save_after == 0):
            #    checkpoint.save(file_prefix=checkpoint_prefix)

