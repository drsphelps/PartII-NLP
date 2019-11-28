import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from cross_validation import train_doc2vec

def get_file_details(root_dir):
    all_files = list()
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))

    return all_files

def prep_files(file_list):
    for i in range(0, len(file_list)):
        f = file_list[i]
        with open(f) as doc:
            line = doc.readline()
            tokens = simple_preprocess(line)
            yield TaggedDocument(tokens, [i])


train_corpus = None

def get_files():
    training_files = get_file_details('../aclImdb_v1')
    global train_corpus
    train_corpus = None
    train_corpus = list(prep_files(training_files))

def build_model(parameters):
    return Doc2Vec(seed=0, dm=parameters['dm'], vector_size=parameters['vector_size'],
                   min_count=parameters['vector_count'], epochs=parameters['epochs'], workers=8,
                   hs=parameters['hs'], window=parameters['window'])

def create_doc2vec_model(f, parameters):
    get_files()

    model = build_model(parameters)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    model.save(f)
    return model

def small_grid_search():
    parameters = {
        'dm':0,
        'vector_size':120,
        'vector_count':2,
        'epochs':10,
        'hs':1,
        'window':20
    }  

    # dms = [1 for x in range(0, 5)] + [1 in range(0, 20)]
    vector_counts = [x%5 for x in range(0, 10)]
    hs = [x%5 for x in range(0, 10)]
    windows = [x for x in range(15, 25)]

    # create_doc2vec_model('0.modelFile', parameters)
    # result = train_doc2vec('0.modelFile')
    # print(parameters, result)
    for x in range(0, 25):
        filename = str(x+1) + '.modelFile'
        # parameters['dm'] = dms[x]
        parameters['vector_count'] = vector_counts[x]
        parameters['hs'] = hs[x]
        parameters['window'] = windows[x]
        create_doc2vec_model(filename, parameters)
        result = train_doc2vec(filename)
        print(parameters, result)

small_grid_search()

# ({'epochs': 10, 'dm': 0, 'vector_count': 0, 'vector_size': 120, 'window': 0, 'hs': 0}, 0.845)
# ({'epochs': 10, 'dm': 0, 'vector_count': 1, 'vector_size': 120, 'window': 1, 'hs': 1}, 0.875)
# ({'epochs': 10, 'dm': 0, 'vector_count': 2, 'vector_size': 120, 'window': 2, 'hs': 2}, 0.85)
# ({'epochs': 10, 'dm': 0, 'vector_count': 3, 'vector_size': 120, 'window': 3, 'hs': 3}, 0.84)
# ({'epochs': 10, 'dm': 0, 'vector_count': 4, 'vector_size': 120, 'window': 4, 'hs': 4}, 0.835)
# ({'epochs': 10, 'dm': 0, 'vector_count': 0, 'vector_size': 120, 'window': 5, 'hs': 0}, 0.845)
# ({'epochs': 10, 'dm': 0, 'vector_count': 1, 'vector_size': 120, 'window': 6, 'hs': 1}, 0.86)
# ({'epochs': 10, 'dm': 0, 'vector_count': 2, 'vector_size': 120, 'window': 7, 'hs': 2}, 0.84)
# ({'epochs': 10, 'dm': 0, 'vector_count': 3, 'vector_size': 120, 'window': 8, 'hs': 3}, 0.82)
# ({'epochs': 10, 'dm': 0, 'vector_count': 4, 'vector_size': 120, 'window': 9, 'hs': 4}, 0.84)
# ({'epochs': 10, 'dm': 0, 'vector_count': 0, 'vector_size': 120, 'window': 10, 'hs': 0}, 0.825)
# ({'epochs': 10, 'dm': 0, 'vector_count': 1, 'vector_size': 120, 'window': 11, 'hs': 1}, 0.88)
# ({'epochs': 10, 'dm': 0, 'vector_count': 2, 'vector_size': 120, 'window': 12, 'hs': 2}, 0.825)
# ({'epochs': 10, 'dm': 0, 'vector_count': 3, 'vector_size': 120, 'window': 13, 'hs': 3}, 0.81)
# ({'epochs': 10, 'dm': 0, 'vector_count': 4, 'vector_size': 120, 'window': 14, 'hs': 4}, 0.825)
# ({'epochs': 10, 'dm': 0, 'vector_count': 0, 'vector_size': 120, 'window': 15, 'hs': 0}, 0.835)
# ({'epochs': 10, 'dm': 0, 'vector_count': 1, 'vector_size': 120, 'window': 16, 'hs': 1}, 0.855)
# ({'epochs': 10, 'dm': 0, 'vector_count': 2, 'vector_size': 120, 'window': 17, 'hs': 2}, 0.82)