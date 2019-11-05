from review_loader import config
from math import log

def calculate_smoothed_log_probs(training_set):
    frequencies = {}
    log_probs = {}
    total_pos = 0
    total_neg = 0

    for (sentiment, filename, features) in training_set:
        is_positive = (sentiment == 'POS')

        if config["presence"]:
            features = list(set(features))

        for feature in features:
            if is_positive:
                total_pos += 1
            else:
                total_neg += 1

            freqs = frequencies.get(feature)
            if freqs is None:
                total_pos += 1
                total_neg += 1
            (pos_freq, neg_freq) = (1, 1) if freqs is None else freqs

            if is_positive:
                frequencies[feature] = (pos_freq + 1, neg_freq)
            else:
                frequencies[feature] = (pos_freq, neg_freq + 1)

    for feature in frequencies:
        (positiveFreq, negativeFreq) = frequencies[feature]
        log_pos = log(float(positiveFreq) / float(total_pos))
        log_neg = log(float(negativeFreq) / float(total_neg))
        log_probs[feature] = (log_pos, log_neg)

    return log_probs

def class_probs(trainingSet):
    pos_count = 0
    neg_count = 0

    for (sentiment, file, features) in trainingSet:
        if sentiment == "POS":
            pos_count += 1
        else:
            neg_count += 1

    return (float(pos_count)/float(pos_count+neg_count), float(neg_count)/float(pos_count+neg_count))


def naiveBayes(trainingSet, testSet):
    feature_probs = calculate_smoothed_log_probs(trainingSet)
    class_probs = class_probs(trainingSet)

    predictions = []

    for (sentiment, file, features) in testSet:
        pos_prob = log(class_probs[0])
        neg_prob = log(class_probs[1])

        for s in features:
            if s not in feature_probs:
                continue

            (pos_feature_prob, neg_feature_prob) = feature_probs[s]

            pos_prob += pos_feature_prob
            neg_prob += neg_feature_prob

        if (pos_prob >= neg_prob):
            predictions.append(("POS", sentiment, file))
        else:
            predictions.append(("NEG", sentiment, file))

    return predictions
