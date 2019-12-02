from review_loader import config
import subprocess


def create_feature_set(data):
    features = {}
    index = 1

    for d in data:
        print(index)
        for feature in d[2]:
            if feature not in features:
                features[feature] = index
                index += 1

    return features


def features_to_svm(features, data_set):
    document_strings = []

    for doc in data_set:
        feature_vector = []
        document_features = {}

        for f in doc[2]:
            if config["presence"]:
                document_features[f] = 1
            else:
                document_features[f] = 1 if f not in document_features else document_features[f] + 1

        for k, v in document_features.items():
            feature_vector.append((features[k], v))

        feature_vector.sort(key=lambda x: x[0])
        feature_vector = ' '.join(map(lambda x: str(x[0])+':'+str(x[1]), feature_vector))

        if doc[0] == 'POS':
            feature_vector = '+1 ' + feature_vector
        else:
            feature_vector = '-1 ' + feature_vector

        document_strings.append(feature_vector)

    return document_strings


def svm_classify(training, test):
    features = create_feature_set(training + test)
    training_strings = features_to_svm(features, training)
    test_strings = features_to_svm(features, test)

    with open('training.data', 'w') as training_file:
        training_file.write('\n'.join(training_strings))
    with open('test.data', 'w') as test_file:
        test_file.write('\n'.join(test_strings))

    subprocess.call("./svm_learn training.data model.data", shell=True)
    subprocess.call("./svm_classify test.data model.data predictions.data", shell=True)

    predictions = []

    with open('predictions.data', 'r') as f:
        predictions.extend([float(p) for p in f.read().split('\n') if p != ""])

    final_predictions = []
    i = 0
    for (sentiment, fileName, features) in test:
        final_predictions.append(("POS" if predictions[i] > 0 else "NEG", sentiment, fileName))
        i += 1

    return final_predictions
