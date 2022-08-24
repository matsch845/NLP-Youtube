import os

import numpy as np
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from nltk.corpus import stopwords

print(common_texts)
base_path = 'out'


def get_flat_words(file_path):
    return_array = []
    with open(file_path) as f:
        for line in f:

            for sentence in line.split('/'):
                sentence_array = []
                for word in sentence.split():
                    if is_word_stop_word(word):
                        continue

                    sentence_array.append(word)

                return_array.append(sentence_array)

    return return_array


def is_word_stop_word(word):
    stop_words = set(stopwords.words('german'))
    additional_stop_words = ['\\', ';', '{', '}', '[', ']', '*']

    if word not in stop_words:
        if not any(asw in word for asw in additional_stop_words):
            return False

    return True


def create_word_2_vec():
    words = np.empty(1)

    for f in os.listdir(base_path):
        # words = words.append(get_flat_words(base_path + "/" + f))
        words = np.concatenate((words, get_flat_words(base_path + "/" + f)))

    word_array = []

    for array in words:
        if type(array) is float or len(array) == 0:
            continue

        word_array.append(array)

    model = Word2Vec(sentences=word_array, vector_size=100, window=5, min_count=1, workers=4)

    # print(model.wv['auto'])
    print(model.wv.most_similar('auto', topn=10))
