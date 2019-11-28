from sklearn.manifold import TSNE
from doc2vec_classification import get_words
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import gensim
import glob

data_file = "./d2v_vec.data"

pos_dir = "../data/POS/*.tag"
neg_dir = "../data/NEG/*.tag"

docModel = gensim.models.Doc2Vec.load("./modelFile")

def get_vecs_from_file_list(files, c):
    vectors = []
    colours = []
    for f in files:
        words = get_words(f)
        doc_vector = docModel.infer_vector(words)
        vectors.append(doc_vector)
        colours.append(c)
    return vectors, colours

def documents_to_vector_dataset():
    pos_files = glob.glob(pos_dir)
    neg_files = glob.glob(neg_dir)

    pos_vectors, pos_colours = get_vecs_from_file_list(pos_files, 1)
    neg_vectors, neg_colours = get_vecs_from_file_list(neg_files, 0)
    
    data = {"p_v" : pos_vectors,  
            "p_c" : pos_colours, 
            "p_fs" : pos_files,
            "n_v" : neg_vectors,
            "n_c" : neg_colours,
            "n_fs" : neg_files,}

    output = open(data_file, 'wb')
    pickle.dump(data, output)
    output.close()

def make_tsne_plot():
    pkl_file = open(data_file, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()   

    vecs = data['p_v'] + data['n_v']
    colours = data['p_c'] + data['n_c']

    tsne_model = TSNE(learning_rate=100, n_components=2, perplexity=5, verbose=2)

    tsne = tsne_model.fit_transform(vecs)

    plt.figure(figsize=(10, 5))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=colours)

    plt.savefig('plot.png')

# documents_to_vector_dataset()
make_tsne_plot()
