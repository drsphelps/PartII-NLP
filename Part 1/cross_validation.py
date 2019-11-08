import naive_bayes
import time
import svm_classifier
from review_loader import *
import statistics

number_of_partitions = 10


def rr_split(features):
    pos_features = features[0:1000]
    neg_features = features[1000:2000]

    partitions = []

    for i in range(0, number_of_partitions):
        partitions.append([])

    for i in range(0, len(pos_features)):
        partition = i % number_of_partitions
        partitions[partition].append(pos_features[i])
        partitions[partition].append(neg_features[i])

    return partitions


def cross_validate(function):
    features = get_features_from_all()
    partitions = rr_split(features)
    results = []

    for test_index in range(0, number_of_partitions):
        training = []
        test = []

        for i in range(0, number_of_partitions):
            if i == test_index:
                test = partitions[i]
            else:
                training.extend(partitions[i])

        classification = function(training, test)

        correct = 0

        for i in range(0, len(classification)):
            if classification[i][0] == classification[i][1]:
                correct += 1

        results.append(float(correct)/float(len(classification)))

    return results


def cross_validate_accuracy(function):
    return statistics.mean(cross_validate(function))


if __name__ == '__main__':
    print("Starting...")
    results = {}

    print(cross_validate_accuracy(svm_classifier.svm_classify))
    time.sleep(2)
    print(cross_validate_accuracy(svm_classifier.svm_classify))
    # results["Uni-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    # config['bigrams'] = True
    # results["Bi-NB"] = cross_validate_accuracy(naive_bayes.naiveBayes)
    # results["Bi-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    # config['bigrams'] = False
    # config['unigram_bigrams'] = True
    # results["UniBi-NB"] = cross_validate_accuracy(naive_bayes.naiveBayes)
    # results["UniBi-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    # config['unigram_bigrams'] = False
    # config['presence'] = True
    # results["PresUni-NB"] = cross_validate_accuracy(naive_bayes.naiveBayes)
    # results["PresUni-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    # config['bigrams'] = True
    # results["PresBi-NB"] = cross_validate_accuracy(naive_bayes.naiveBayes)
    # results["PresBi-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    # config['bigrams'] = False
    # config['unigram_bigrams'] = True
    # results["PresUniBi-NB"] = cross_validate_accuracy(naive_bayes.naiveBayes)
    # results["PresUniBi-SVM"] = cross_validate_accuracy(svm_classifier.svm_classify)
    #
    # print(results)
    #
    # '''{'Uni-SVM': 0.7375, 'PresUni-SVM': 0.8574999999999999, 'Bi-SVM': 0.7805, 'PresUniBi-SVM': 0.8720000000000001, 'Bi-NB': 0.8445, 'Uni-NB': 0.8125, 'PresBi-SVM': 0.835, 'UniBi-SVM': 0.7474999999999999, 'PresUniBi-NB': 0.8210000000000001, 'UniBi-NB': 0.833, 'PresBi-NB': 0.8480000000000001, 'PresUni-NB': 0.7215}'''
