import json
import spacy
import time
from gensim.models import Word2Vec
from utils.preprocessing import *
from nltk.tokenize import word_tokenize

nlp = spacy.load("en_core_web_sm")

file_path_train = 'data/train-v1.1.json'
file_path_dev = 'data/dev-v1.1.json'

with open(file_path_train, "r", encoding='utf-8') as reader:
    json_train = json.load(reader)["data"]

with open(file_path_dev, "r", encoding='utf-8') as reader:
    json_dev = json.load(reader)["data"]

def preprocess_json(input_data):
    ids2context = {}
    dataset = []

    data_id = 0
    context_id = 0


    for data in input_data:
        title = data["title"]
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            #context_id = uuid.uuid4()
            ids2context[context_id] = context
            for qa in paragraph["qas"]:
                qa_dict = {}
                qa_dict["question"] = qa["question"]
                qa_dict["context_id"] = context_id
                qa_dict["title"] = title
                qa_dict["data_id"] = data_id
                for a in qa["answers"]:
                    qa_dict["answer_start"] = a["answer_start"]
                    qa_dict["answer"] = a["text"]
                dataset.append(qa_dict)
                data_id = data_id + 1
            context_id = context_id + 1

    return (dataset, ids2context)

squad_train, ids2context_train = preprocess_json(json_train)
squad_dev, ids2context_dev = preprocess_json(json_dev)

#ids = [data["data_id"] for data in squad_train]

print(ids2context_train[0])

start = time.time()

doc = nlp(ids2context_train[0])

print(time.time() - start)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)



