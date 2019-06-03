from trainers.trainer import Trainer

class TrainerTransformer(Trainer):
    def __init__(self, config, vocab):
        super(TrainerTransformer, self).__init__(config, vocab)

    def train(self, model, dataset_train, dataset_test):
        pass