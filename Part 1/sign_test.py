from math import ceil
from cross_validation import get_features_from_all, rr_split
from scipy.stats import binom
import naive_bayes
import svm_classifier
from review_loader import config


def sign_test(classifier_a, classifier_b):

    plus = 0
    neg = 0
    null = 0

    for i in range(0, len(classifier_a)):
        (pred1, sent1, file1) = classifier_a[i]
        (pred2, sent2, file2) = classifier_b[i]

        print(pred1, pred2, sent1, file1 == file2)

    classifier_a = [sentiment1 == prediction1 for (prediction1, sentiment1, filename1) in classifier_a]
    classifier_b = [sentiment2 == prediction2 for (prediction2, sentiment2, filename2) in classifier_b]

    for (pred_a, pred_b) in zip(classifier_a, classifier_b):
        if pred_a and not pred_b:
            plus += 1
        elif pred_b and not pred_a:
            neg += 1
        else:
            null += 1

    q = 0.5
    N = int(2 * (ceil(float(null) / 2.0)) + plus + neg)
    k = int(ceil(float(null) / 2.0) + min(plus, neg))

    print(plus, neg, null)

    return 2 * binom.cdf(k, N, q)


def aggregated_cross_val(function):
    features = get_features_from_all()
    partitions = rr_split(features)
    full = []

    for x in range(0, 10):
        training = []
        test = []

        for i in range(0, 10):
            if i == x:
                test = partitions[i]
            else:
                training.extend(partitions[i])

        classification_a = function(training, test)
        full.extend(classification_a)
    return full

results = {}

a_full = aggregated_cross_val(naive_bayes.naiveBayes)
config['presence'] = True
config['unigram_bigrams'] = True
b_full = aggregated_cross_val(naive_bayes.naiveBayes)
results['NB'] = sign_test(b_full, a_full)


config['presence'] = False
config['unigram_bigrams'] = False
a_full = aggregated_cross_val(svm_classifier.svm_classify)
config['presence'] = True
config['unigram_bigrams'] = True
b_full = aggregated_cross_val(svm_classifier.svm_classify)
results['SVM'] = sign_test(b_full, a_full)

print(results)
# {'NB': 0.08929648432127088, 'SVM': 0.3253010708558339}