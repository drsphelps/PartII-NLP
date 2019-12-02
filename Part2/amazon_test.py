import json
import random
import svmlight
from doc2vec_classification import predict
from review_loader import get_features_from_all
from cross_validation import amazon_test, rr_split
from gensim.utils import simple_preprocess
import string 

def create_feature_set(features, data, index):
    for d in data:
        for feature in d[index]:
            if feature not in features:
                features[feature] = len(features) + 1

    return features

def create_svm_vectors(data, all_features, index):
    training = []
    for d in data:
        vector = []
        for feature in d[index]:
            vector.append((all_features[feature], 1))
        vector = list(set(vector))
        vector.sort()
        training.append((1 if d[0] == 'POS' else -1, vector))
    return training, all_features

# Had to reimplement as don't have function that returns a model
# Also allows me to use a library unlike in task 1
def create_smv_model():
    features = get_features_from_all()
    partitions = rr_split(features)
    training = []
    for i in range(1, len(partitions)):
        training.extend(partitions[i])

    features = create_feature_set({}, training, 2)
    svm_vectors, all_features = create_svm_vectors(training, features, 2)
    model = svmlight.learn(svm_vectors, type='classification')
    return model, all_features

def format_reviews(reviews, features):
    reviews = [(s, simple_preprocess(f)) for (s,f) in reviews]
    features = create_feature_set(features, reviews, 1)
    vectors, features = create_svm_vectors(reviews, features, 1)
    return vectors, features


def get_amazon_reviews(path):
    negative_reviews = []
    positive_reviews = []

    with open(path) as json_file:
        line = json_file.readline()
        while line:
            amazon_data = json.loads(line)
            if int(amazon_data['overall']) == 1:
                negative_reviews.append(amazon_data)
            if int(amazon_data['overall']) == 5:
                positive_reviews.append(amazon_data)
            line = json_file.readline()

    reviews = random.sample(negative_reviews, 100) + random.sample(positive_reviews, 100)
    reviews = [('POS' if int(d['overall']) == 5 else 'NEG', d['reviewText']) for d in reviews]

    return reviews

doc2vec_svm = amazon_test()
svm_model, features = create_smv_model()

d_a = []
s_a = []

for i in range(0, 10):
    rs = get_amazon_reviews('amazon_data/Grocery_and_Gourmet_Food_5.json')

    reviews, features = format_reviews(rs, features)
    predictions = svmlight.classify(svm_model, reviews)
    correct = 0.0
    total = 0.0
    for i in range(len(reviews)):
        prediciton = 1 if predictions[i] >= 0 else -1
        if prediciton == reviews[i][0]:
            correct += 1.0
        total += 1.0
    s_a.append(correct/total)


    results = predict('./modelFile', doc2vec_svm, rs, False)
    total = 0.0
    correct = 0.0
    for i in range(0, len(reviews)):
        total += 1.0
        result = results[rs[i][1]] == rs[i][0]
        if result:
            correct += 1.0
    d_a.append(correct/total)

print(s_a)
print("SVM average: " + str(sum(s_a)/float(len(s_a))))
print(d_a)
print("D2V average: " + str(sum(d_a)/float(len(d_a))))