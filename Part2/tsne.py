from sklearn.manifold import TSNE
from doc2vec_classification import get_words
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle
import gensim
import glob
import nltk
from numpy import array

data_file = "./d2v_vec.data"

pos_dir = "../data/POS/*.tag"
neg_dir = "../data/NEG/*.tag"

docModel = gensim.models.Doc2Vec.load("./modelFile")

tags = {
"CD" :  "black"
,"CC" :  "black"
,"DT" :  "black"
,"EX" :  "black"
,"FW" :  "slategrey"
,"IN" :  "gold"
,"JJR" :  "green"
,"JJS" : "green"
,"JJ" : "green"
,"LS" :  "black"
,"MD" :  "red"
,"NN" :  "blue"
,"NNS" : "blue"
,"NNP" : "purple"
,"NNPS" : "purple"
,"PDT" : "black"
,"POS" : "black"
,"PRP" : "darkorange"
,"PRP$" : "darkorange"
,"RB" :  "pink"
,"RBR" : "pink"
,"RBS" : "pink"
,"RP" :  "black"
,"TO" :  "black"
,"UH" :  "peru"
,"VB" :  "red"
,"VBD" : "red"
,"VBN" : "red"
,"VBG" : "red"
,"VBP" : "red"
,"VBZ" : "red"
,"WDT" : "black"
,"WP" :  "darkorange"
,"WP$" : "darkorange"
,"WRB" : "pink"}

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
    

    posFileNames = ["".join(file.split("_")[0].split("/")[2:4]) for file in pos_files]
    negFileNames = ["".join(file.split("_")[0].split("/")[2:4])for file in neg_files]

    data = {"p_v" : pos_vectors,  
            "p_c" : pos_colours, 
            "p_fs" : pos_files,
            "p_fn" :posFileNames,
            "n_v" : neg_vectors,
            "n_c" : neg_colours,
            "n_fs" : neg_files,
            "n_fn": negFileNames}

    output = open(data_file, 'wb')
    pickle.dump(data, output)
    output.close()

def get_dataset():
    pickie = open(data_file, 'rb')
    data = pickle.load(pickie)
    pickie.close()
    return data

def make_tsne_plot():
    data = get_dataset()

    limit = -1

    vecs = data['p_v'][:limit] + data['n_v'][:limit]
    colours = data['p_c'][:limit] + data['n_c'][:limit]

    tsne_model = TSNE(learning_rate=100, n_components=2, perplexity=5, verbose=2)

    tsne = tsne_model.fit_transform(vecs)

    plt.figure(figsize=(10, 5))
    plt.scatter(tsne[:, 0], tsne[:, 1], c=colours)

    plt.savefig('plot.png')

def tnse_n_doc():
    data = get_dataset()

    n = 40
    docnum = 1

    vecs = data["p_v"][:docnum] + data["n_v"][:docnum]

    wordVecs = []
    words = []
    wordTypes = []
    for i in range(docnum):
        wordList = get_words(data["p_fs"][i]) + get_words(data["n_fs"][i])
        for word in wordList:
            try:
                wordVec = docModel.wv.get_vector(word)
            except(KeyError):
                continue
            #wordVec = docModel.infer_vector([str(word)])
            if word not in words:  
                tag = nltk.pos_tag([word])[0][1]
                if tag in ["JJR","JJS","JJ"]:
                    words.append(word)
                    wordVecs.append(wordVec)
                    if i <= docnum/2:
                        print(1)
                        wordTypes.append(1)
                    else:
                        print(0)
                        wordTypes.append(0)

    X_tsne = TSNE(n_components=2, perplexity=5, verbose=2).fit_transform(vecs + wordVecs)

    fig, ax = plt.subplots()
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=array(["silver" for i in range(2*docnum)]+wordTypes))

    for i, txt in enumerate(data["p_fn"][:docnum] + data["n_fn"][:docnum] + words):
        ax.annotate(txt, (X_tsne[i, 0], X_tsne[i, 1]))

    plt.savefig('plot.png')


# documents_to_vector_dataset()
tnse_n_doc()
