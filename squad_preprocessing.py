import json
import spacy
import pickle
from tqdm import tqdm
from vocabulary import Vocabulary

vocab = Vocabulary()

nlp = spacy.load("en_core_web_sm")

file_path_json_train = 'data/train-v1.1.json'
file_path_json_dev = 'data/dev-v1.1.json'

folder_path_train = "data/train/"
folder_path_dev = "data/dev/"

with open(file_path_json_train, "r", encoding='utf-8') as reader:
    json_train = json.load(reader)["data"]

with open(file_path_json_dev, "r", encoding='utf-8') as reader:
    json_dev = json.load(reader)["data"]


def preprocess_text(text):
    doc = nlp(text)
    lemma_tokens = []
    for token in doc:
        lemma_tokens.append(token.lemma_)
    vocab.add_sentence_to_vocab(lemma_tokens)
    return(doc)

def preprocess_json(input_data, folder_path):
    data_id = 0
    for data in tqdm(input_data):
        for paragraph in data["paragraphs"]:
            paragraph["context"] = preprocess_text(paragraph["context"])
            for qa in paragraph["qas"]:
                qa["question"] = preprocess_text(qa["question"])
                for a in qa["answers"]:
                    a["text"] = preprocess_text(a["text"])
        with open(folder_path + str(data_id) + ".pickle", 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        data_id = data_id + 1

preprocess_json(json_train, folder_path_train)
preprocess_json(json_dev, folder_path_dev)

vocab.save_vocab()



'''
def preprocess_text(text):
    #doc = nlp(text)
    #lemma_tokens = []
    #for token in doc:
    #    lemma_tokens.append(token.lemma_)
    #vocab.add_sentence_to_vocab(lemma_tokens)
    #return(doc)
    return("foo")

def preprocess_json(input_data):

    ids2context = {}
    dataset = {}

    data_id = 0
    context_id = 0

    for data in tqdm(input_data):
        title = data["title"]
        for paragraph in data["paragraphs"]:
            ids2context[context_id] = preprocess_text(paragraph["context"])
            for qa in paragraph["qas"]:
                qa_dict = {}
                qa_dict["title"] = title
                qa_dict["context_id"] = context_id

                qa_dict["question"] = preprocess_text(qa["question"])

                for a in qa["answers"]:
                    qa_dict["answer_start"] = a["answer_start"]
                    qa_dict["answer"] = preprocess_text(a["text"])
                dataset[data_id] = qa_dict
                data_id = data_id + 1

            context_id = context_id + 1

    return (dataset, ids2context)

squad_train, ids2context_train = preprocess_json(json_train)
squad_dev, ids2context_dev = preprocess_json(json_dev)

vocab.save_vocab()


#with open("data/squad_train", 'wb') as handle:
#    pickle.dump(squad_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open("data/ids2context_train", 'wb') as handle:
#    pickle.dump(ids2context_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open("data/squad_dev", 'wb') as handle:
#    pickle.dump(squad_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open("data/ids2context_dev", 'wb') as handle:
#    pickle.dump(ids2context_dev, handle, protocol=pickle.HIGHEST_PROTOCOL)
'''