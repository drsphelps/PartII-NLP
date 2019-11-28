import os

config = {
    "stem": True,
    "bigrams": False,
    "unigram_bigrams": False,
    "presence": False,
}

pos_dir = "./../data/POS/STEMMED" if config["stem"] else "./../data/POS"
neg_dir = "./../data/NEG/STEMMED" if config["stem"] else "./../data/NEG"

file_ending = ".tag_STM" if config['stem'] else ".tag"

def get_unigrams_from_file(path):
    features = []
    with open(path) as file:
        features.extend([f.lower() for f in file.read().split("\n") if f != ''])
    return features


def get_bigrams_from_file(path):
    unigrams = get_unigrams_from_file(path)

    bigrams = []

    for i in range(1, len(unigrams)):
        bigrams.append(unigrams[i-1] + ":" + unigrams[i])

    return bigrams


def get_both_from_file(path):
    features = get_unigrams_from_file(path) + get_bigrams_from_file(path)
    return features


def get_all_files(pos_list, neg_list, extractor):
    features = []

    for f in pos_list:
        features.append(("POS", f, extractor(pos_dir + "/" + f)))
    for f in neg_list:
        features.append(("NEG", f, extractor(neg_dir + "/" + f)))

    return features


def get_features_from_all():
    positiveFileList = sorted(list(filter(lambda s: s.endswith(file_ending), os.listdir(pos_dir))))
    negativeFileList = sorted(list(filter(lambda s: s.endswith(file_ending), os.listdir(neg_dir))))

    features = []

    if config["bigrams"]:
        features = get_all_files(positiveFileList, negativeFileList, get_bigrams_from_file)
    elif config['unigram_bigrams']:
        features = get_all_files(positiveFileList, negativeFileList, get_both_from_file)
    else:
        features = get_all_files(positiveFileList, negativeFileList, get_unigrams_from_file)

    return features
