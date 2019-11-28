from create_model import create_doc2vec_model
from cross_validation import train_doc2vec, test_doc2vec

parameters = {
        'dm':0,
        'vector_size':100,
        'vector_count':2,
        'epochs':20,
        'hs':10,
        'window':2
    }    

def train_model():
    create_doc2vec_model(parameters)
    result = train_doc2vec()
    print(result)
    print(test_doc2vec())

train_model()
