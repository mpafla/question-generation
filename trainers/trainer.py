
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
        self.prefetch = config["trainer"]["prefetch"]
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

    def diagnostics(self, epoch, batch):
        template = 'Epoch: {}/{}, Batch: {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        diagnostics = template.format(epoch + 1,
                                      self.epochs,
                                      batch,
                                      self.train_loss.result(),
                                      self.train_accuracy.result() * 100,
                                      self.test_loss.result(),
                                      self.test_accuracy.result() * 100)
        return diagnostics


    def train(self, model, dataset_train, dataset_test):
        raise Exception("Train method not implemented")