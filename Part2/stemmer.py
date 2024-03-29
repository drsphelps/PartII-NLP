import os
from nltk.stem import PorterStemmer

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

ps = PorterStemmer()
pos_folder = "./../data/POS/"
neg_folder = "./../data/NEG/"

pos_files = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(pos_folder))))
neg_files = sorted(list(filter(lambda s: s.endswith(".tag"), os.listdir(neg_folder))))


def stem_files(file_list, folder):
    for f in file_list:
        with open(folder + f, "r") as read_file:
            with open(folder + "STEMMED/" + f + "_STM", "w") as write_file:
                read_line = read_file.readline()
                while read_line:
                    line = ps.stem(read_line.strip()) + '\n'
                    write_file.write(line)
                    read_line = read_file.readline()


stem_files(pos_files, pos_folder)
stem_files(neg_files, neg_folder)
