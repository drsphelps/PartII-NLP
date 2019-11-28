import random
from cross_validation import test_doc2vec
from sign_test import aggregated_cross_val
from naive_bayes import naiveBayes
from svm_classifier import svm_classify

def calculate_accuracy(lst):
    values = lst.values()
    return float(values.count(True))/float(len(values))

def permutation(pred1, pred2, R):
    s = 0.0
    mean_difference = abs(calculate_accuracy(pred1) - calculate_accuracy(pred2))

    print(pred1.keys())
    print(pred2.keys())

    keys = pred2.keys()
    if len(pred1) < len(pred2):
        keys = pred1.keys()

    for _ in range(R):
        new1 = dict(pred1)
        new2 = dict(pred2)


        for i in range(len(keys)):
            if random.randint(0, 1) == 1:
                temp = new1[keys[i]]
                new1[keys[i]] = new2[keys[i]]
                new2[keys[i]] = temp
        new_difference = abs(calculate_accuracy(new1) - calculate_accuracy(new2))
        if new_difference >= mean_difference:
            s+=1.0
    
    return (s+1.0)/(float(R)+1)

def perm_test():
    nb_results = aggregated_cross_val(naiveBayes)
    nb_dict = {}
    for (sen1, sen2, f) in nb_results:
        nb_dict[f[:-4]] = sen1 == sen2
    # svm_results = aggregated_cross_val(svm_classify)
    # svm_dict = {}
    # for (sen1, sen2, f) in svm_results:
    #     svm_dict[f] = sen1 == sen2
    d2v_dict = test_doc2vec()

    r = 5000

    # print(permutation(nb_dict, svm_dict, r))
    print(permutation(nb_dict, d2v_dict, r))
    # print(permutation(svm_dict, d2v_dict, r))
    
perm_test()