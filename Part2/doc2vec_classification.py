import os
import svmlight
import subprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

def get_words(path, f=True):
    if f:
        with open(path,'r') as f:
            words = simple_preprocess(f.read())
    else:
        words = simple_preprocess(path)
    return words

def train_svm(f, training):
    d2v_model = Doc2Vec.load(f)
    f_training = []
    for (sentiment, path) in training:
        words = get_words(path)
        vector = [(i+1, p) for i, p in enumerate(d2v_model.infer_vector(words))]
        f_training.append((1 if sentiment == 'POS' else -1, vector))
    
    return svmlight.learn(f_training, type='classification')

def predict(f, svm_model, test, w=True):
    d2v_model = Doc2Vec.load(f)
    results = {}

    f_test = []
    for (_, path) in test:
        words = get_words(path, w)
        vector = [(i+1, p) for i, p in enumerate(d2v_model.infer_vector(words))]
        f_test.append((0, vector))

    predictions = svmlight.classify(svm_model, f_test)

    for i in range(len(predictions)):
        prediction = "POS" if predictions[i] > 0 else "NEG"
        results[test[i][1]] = prediction

    return results

