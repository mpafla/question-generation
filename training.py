import sys
import os
import yaml
import time
import numpy as np

from vocabulary import Vocabulary
from dataset_creator import DatasetCreator
from embedder import Embedder
from utils.constants import Constants
from tensorflow.python.keras.metrics import CategoricalAccuracy


def main():
    configs_path = sys.argv[1:][0]
    with open(configs_path, 'r') as file:
        config = yaml.safe_load(file)

    vocab = Vocabulary(config)

    dataset_creator = DatasetCreator(vocab, config)
    dataset_train, dataset_dev = dataset_creator.create_datasets()

    embedder = Embedder(vocab, config)

    embeddings_matrix = embedder.get_embeddings_matrix()






    #start = time.time()

    #for i, data in enumerate(dataset_train):
    #   print(data)
    #   if i > 1:
    #       break



    #model_file_name = "models." + config["model"]["file_name"]
    #model_class_name = config["model"]["class_name"]

    #module = __import__(model_file_name, fromlist=[model_class_name])
    #model = getattr(module, model_class_name)(config)



if __name__ == "__main__":
    main()