import os

import nltk
import pandas as pd
from googletrans import Translator
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

from constants import *

nltk.download([
    "names",
    "stopwords",
    "state_union",
    "twitter_samples",
    "movie_reviews",
    "averaged_perceptron_tagger",
    "vader_lexicon",
    "punkt",
])

base_path = 'out'
base_path_translated = 'out/translated'
translator = Translator()
max_view_count = 1800000


def analyze(translate_new):
    df = pd.DataFrame(columns=[('%s' % video_id), ('%s' % view_count), ('%s' % like_count), ('%s' % favorite_count),
                               ('%s' % comment_count), ('%s' % pos_score), ('%s' % neg_score),
                               ('%s' % neu_score), ('%s' % comp_score)])

    for f in os.listdir(base_path):
        if f == "translated" or f == ".DS_Store":
            continue

        if translate_new:
            lines = get_line_array(base_path + "/" + f)

            lines = tokenize_and_filter_line_array(lines)

            lines = [line for line in lines if len(line) > 0]

            lines_as_string = translator.translate(list_to_string(lines, " "), src='de').text

            file = open("out/translated/" + f, "x")
            file.write(lines_as_string)
            file.close()

        split_arr = f.split('%')

        if int(split_arr[2]) > max_view_count:
            continue

        lines = get_line_array(base_path_translated + "/" + f)

        sia = SentimentIntensityAnalyzer()

        positive_score = 0
        negative_score = 0
        neutral_score = 0
        compound_score = 0

        tokenized_words = word_tokenize(lines[0])

        # for line in tokenized_words:
        #     sentiment = sia.polarity_scores(line)
        #     positive_score += sentiment['pos']
        #     negative_score += sentiment['neg']
        #     neutral_score += sentiment['neu']
        #     compound_score += sentiment['compound']
        #
        # positive_score = positive_score / len(tokenized_words)
        # negative_score = negative_score / len(tokenized_words)
        # neutral_score = neutral_score / len(tokenized_words)
        # compound_score = compound_score / len(tokenized_words)

        sentiment = sia.polarity_scores(lines[0])
        positive_score = sentiment['pos']
        negative_score = sentiment['neg']
        neutral_score = sentiment['neu']
        compound_score = sentiment['compound']

        d = {
            video_id: split_arr[1],
            view_count: int(split_arr[2]),
            like_count: int(split_arr[3]),
            favorite_count: int(split_arr[4]),
            comment_count: int(split_arr[5].split('.')[0]),
            pos_score: positive_score,
            neg_score: negative_score,
            neu_score: neutral_score,
            comp_score: compound_score
        }

        df = df.append(d, ignore_index=True)

    df.reset_index().to_csv("features.csv", sep=';')


def tokenize_and_filter_line_array(lines_array):
    return_array = []

    for line in lines_array:
        filtered_words = []

        stop_words = set(stopwords.words('german'))

        words = nltk.word_tokenize(line)

        additional_stop_words = ['\\', ';', '{', '}', '[', ']', '*', '\n']

        for w in words:
            if w not in stop_words:

                if not any(asw in w for asw in additional_stop_words):
                    filtered_words.append(w)

        return_array.append(filtered_words)

    return return_array


def get_line_array(file_path):
    lines_array = []
    with open(file_path) as f:
        try:
            for line in f:
                lines_array.append(line)
        except:
            print(file_path)

    return lines_array


def translate_lines(lines):
    final_list = []

    for line in lines:
        s = ""
        for word in line:
            s = s + " " + word

        final_list.append(translator.translate(s, src='de').text)

    return final_list


def list_to_string(l, sep="\n"):
    s = ""
    for elem in l:
        for word in elem:
            s = s + sep + word

    return s
