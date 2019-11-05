import naive_bayes
import svm_classifier
from review_loader import *
import statistics

number_of_partitions = 3


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


print(statistics.mean(cross_validate(svm_classifier.svm_classify)))
