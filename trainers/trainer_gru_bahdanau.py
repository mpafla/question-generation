
import tensorflow as tf
import os
from utils.constants import Constants
import time
from trainers.trainer import Trainer



class TrainerGRUBahdanau(Trainer):
    def __init__(self, config, vocab):
        super(TrainerGRUBahdanau, self).__init__(config, vocab)

    @tf.function
    def run_through_step(self, encoder, decoder, data, training=True):
        input = data[0]
        target = data[1]
        target_one_hot = data[2]

        #Check if batch size is correct
        if target_one_hot.shape[0] == self.batch_size:

            start = time.time()
            loss = 0
            answers = input[:,0]
            contexts = input[:, 1]

            #targets = tf.math.argmax(target_one_hot, axis=2)
            targets = target

            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder.call([answers, contexts])

                dec_hidden = enc_hidden
                dec_input = tf.expand_dims([self.vocab.get_index_for_token(Constants.SOS)] * target_one_hot.shape[0], 1)


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
                

    def train(self, model, dataset_train, dataset_test):
        dataset_train = dataset_train.batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
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

            print("Loading data")
            for data in dataset_train:
                self.run_through_step(encoder, decoder, data, training=True)
                step = step + 1

                # if (step % 10 == 0):
                #    print(diagnostics(epoch, step))
                print(self.diagnostics(epoch, step))

            print("Getting accuracy from test dataset")
            for data in dataset_test:
                self.run_through_step(encoder, decoder, data, training=False)

            print(self.diagnostics(epoch, step))
            print('Time taken for training this epoch is {} sec'.format(time.time() - start))

            with open("models/diagnostics.txt", "a") as text_file:
                print(self.diagnostics(epoch, step), file=text_file)

            #if (epoch % self.save_after == 0):
            #    checkpoint.save(file_prefix=checkpoint_prefix)

