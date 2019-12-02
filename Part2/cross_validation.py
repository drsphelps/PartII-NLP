import os
import pickle
from doc2vec_classification import train_svm, predict

number_of_partitions = 10
pos_dir = '../data/POS/'
neg_dir = '../data/NEG/'
file_ending = '.tag'


def get_all_files():
    pos = sorted(list(filter(lambda s: s.endswith(file_ending), os.listdir(pos_dir))))
    neg = sorted(list(filter(lambda s: s.endswith(file_ending), os.listdir(neg_dir))))
    return [('POS', pos_dir+p) for p in pos] + [('NEG', neg_dir+n) for n in neg]

def rr_split(features):
    size = len(features) / 2
    pos_features = features[0:size]
    neg_features = features[size:len(features)]

    partitions = []

    for i in range(0, number_of_partitions):
        partitions.append([])

    for i in range(0, len(pos_features)):
        partition = i % number_of_partitions
        partitions[partition].append(pos_features[i])
        partitions[partition].append(neg_features[i])

    return partitions

def remove_first(files):
    partitions = rr_split(files)
    removed = []

    for i in range(0, len(partitions)):
        if i != 0:
            removed.extend(partitions[i])
    return removed

def train_doc2vec(f):
    partitions = rr_split(get_all_files())
    
    training = []
    validation = []
    for i in range(0, len(partitions)):
        if i == 0:
            validation = partitions[i]
        else:
            training.extend(partitions[i])

    svm_model = train_svm(f, training)
    results = predict(f, svm_model, validation)

    total = 0.0
    correct = 0.0

    for i in range(0, len(validation)):
        total += 1.0
        if results[validation[i][1]] == validation[i][0]:
            correct += 1.0

    return correct/total

def test_doc2vec():
    dataset = remove_first(get_all_files())
    positive = [x for x in dataset if x[0] == 'POS']
    negative = [x for x in dataset if x[0] == 'NEG']

    partitions = rr_split(positive + negative)

    complete_results = {}
    r = []

    for test_index in range(0, number_of_partitions):
        training = []
        test = []
        print("test index: " + str(test_index))

        for i in range(0, number_of_partitions):
            if i == test_index:
                test = partitions[i]
            else:
                training.extend(partitions[i])

        svm_model = train_svm('./0.modelFile', training)
        results = predict('./0.modelFile', svm_model, test)

        total = 0.0
        correct = 0.0

        for i in range(0, len(test)):
            total += 1.0
            result = results[test[i][1]] == test[i][0]
            if result:
                correct += 1.0
            complete_results[test[i][1].split('/')[-1]] = result
        r.append(correct/total)

    return complete_results

def amazon_test():
    dataset = remove_first(get_all_files())
    positive = [x for x in dataset if x[0] == 'POS']
    negative = [x for x in dataset if x[0] == 'NEG']
    svm_model = train_svm('./modelFile', positive + negative)


    print(len(positive+negative))

    return svm_model
