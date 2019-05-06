import csv
import pickle

from gensim.models import Word2Vec
from utils.preprocessing import *

questions = []
questions_input = []
answers = []
sources = []

MIN_WORD_APPEARANCES = 10
W2V_DIM = 300


preprocess_functions = [removePunctuation, removeEmptyTokens, lowerCase, removeStopWords]

with open('data/data_cleaned.csv', encoding="utf8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    # Get rid of rowname
    next(reader)

    for row in reader:
        question = word_tokenize(row[0])
        answer = word_tokenize(row[1])
        source = word_tokenize(row[2])

        # Apply every function in preprocess_functions to corpus
        question = reduce(lambda x, y: y(x), preprocess_functions, question)
        answer = reduce(lambda x, y: y(x), preprocess_functions, answer)
        source = reduce(lambda x, y: y(x), preprocess_functions, source)

        questions.append(question)
        answers.append(answer)
        sources.append(source)
print("Data read and preprocessed")


#Embeddings
data = questions + answers + sources


w2v_path = "models/word2vec.model"

try:
    print("Loading word2vec model")
    w2v_model = Word2Vec.load(w2v_path)
    print("Loaded word2vec model")
except:
    print("Loading of word2vec failed")
    print("Generating new word2vec model")
    w2v_model = Word2Vec(data, size=W2V_DIM, window=5, min_count=MIN_WORD_APPEARANCES, workers=4)
    print("word2vec generated")
    w2v_model.save(w2v_path)

vocab = w2v_model.wv.vocab
vocabulary_size = len(vocab)
print("Vocabulary size: {}".format(vocabulary_size))

questions_indexed = getIndexesForCorpus(questions, w2v_model)
answers_indexed = getIndexesForCorpus(answers, w2v_model)
sources_indexed = getIndexesForCorpus(sources, w2v_model)
questions_input_indexed = []

for i, (question, answer, source) in enumerate(zip(questions_indexed, answers_indexed, sources_indexed)):

    #Copy question to create right-shifted input to decoder (by adding SOS)
    question_input = addSOS(question.copy(), vocabulary_size)
    questions_input_indexed.append(question_input)

    question = addEOS(question, vocabulary_size)
    answer = addEOS(answer, vocabulary_size)
    source = addEOS(source, vocabulary_size)



    if (len(answer) < len(source)):
        padSequence(answer, len(source), vocabulary_size)
    elif (len(answer) > len(source)):
        padSequence(source, len(answer), vocabulary_size)

    #if(len(question_input) is not len(question)):


print("Data prepared")

print(len(questions_indexed[0]))
print(len(questions_input_indexed[0]))
print(len(answers_indexed[0]))
print(len(sources_indexed[0]))


with open("data/questions_indexed.pickle", 'wb') as handle:
    pickle.dump(questions_indexed, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/answers_indexed.pickle", 'wb') as handle:
    pickle.dump(answers_indexed, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/sources_indexed.pickle", 'wb') as handle:
    pickle.dump(sources_indexed, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/questions_input_indexed.pickle", 'wb') as handle:
    pickle.dump(questions_input_indexed, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Data (indexed) saved")

embedding_matrix = createEmbeddingsMatrix(vocabulary_size, W2V_DIM, w2v_model)

with open("data/embedding_matrix.pickle", 'wb') as handle:
    pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Embeddings matrix saved")




