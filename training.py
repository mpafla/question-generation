import sys
import yaml


from vocabulary import Vocabulary
from embedder import Embedder
from data_preprocessor import DataPreprocessor



def main():
    configs_path = sys.argv[1:][0]
    with open(configs_path, 'r') as file:
        config = yaml.safe_load(file)

    vocab = Vocabulary(config)

    data_preprocessor = DataPreprocessor(vocab, config)
    dataset_train, dataset_dev = data_preprocessor.create_datasets()

    embedder = Embedder(vocab, config)
    embeddings_matrix = embedder.get_embeddings_matrix()

    model_file_name = config["model"]["file_name"]
    model_class_name = config["model"]["class_name"]

    module = __import__(model_file_name, fromlist=[model_class_name])
    model = getattr(module, model_class_name)(config, embeddings_matrix, vocab)

    trainer_file_name = config["trainer"]["file_name"]
    trainer_class_name = config["trainer"]["class_name"]

    module = __import__(trainer_file_name, fromlist=[trainer_class_name])
    trainer = getattr(module, trainer_class_name)(config, vocab)


    trainer.train(model, dataset_train, dataset_dev)



if __name__ == "__main__":
    main()