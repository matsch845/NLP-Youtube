import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras import backend as K
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import StandardScaler
import random

from constants import *

training_split = 0.85
randomlist = random.sample(range(0, 70), 5)


def get_training_data(file):
    return pd.read_csv(file, header=0, delimiter=";")


# https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/
def train_deep_regression_model(file):
    df_test_x, df_test_y, df_training_x, df_training_y = prepare_data(file)

    sc = StandardScaler()
    df_training_x = sc.fit_transform(df_training_x)
    df_test_x = sc.transform(df_test_x)

    input_layer = Input(shape=(df_training_x.shape[1],))
    dense_layer_1 = Dense(100, activation='relu')(input_layer)
    dense_layer_2 = Dense(100, activation='relu')(dense_layer_1)
    dense_layer_3 = Dense(100, activation='relu')(dense_layer_2)
    dense_layer_4 = Dense(100, activation='relu')(dense_layer_3)
    dense_layer_5 = Dense(100, activation='relu')(dense_layer_4)
    dense_layer_6 = Dense(100, activation='relu')(dense_layer_5)
    dense_layer_7 = Dense(100, activation='relu')(dense_layer_6)
    dense_layer_8 = Dense(100, activation='relu')(dense_layer_7)
    dense_layer_9 = Dense(50, activation='relu')(dense_layer_8)
    dense_layer_10 = Dense(25, activation='relu')(dense_layer_9)
    output = Dense(1)(dense_layer_10)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(loss="mean_squared_error", optimizer="adam",
                  metrics=["mean_squared_error", "mean_squared_logarithmic_error"])

    history = model.fit(df_training_x, df_training_y, batch_size=2, epochs=100, verbose=1, validation_split=0.2)

    print("-----History-----")
    print(history.history)

    pred_train = model.predict(df_training_x)
    pred = model.predict(df_test_x)

    print(np.sqrt(mean_squared_error(df_training_y, pred_train)))
    print(np.sqrt(mean_squared_error(df_test_y, pred)))

    # from https://stackoverflow.com/questions/42351184/how-to-calculate-r2-in-tensorflow
    unexplained_error = tf.reduce_sum(tf.square(tf.subtract(df_test_y, pred)))
    total_error = tf.reduce_sum(tf.square(tf.subtract(df_test_y, tf.reduce_mean(df_test_y))))
    R_squared = tf.subtract(1, tf.divide(unexplained_error, total_error))

    print("r2 score: " + str(R_squared))

    print_predictions(pred, df_test_x, df_test_y)


def print_predictions(pred, df_test_x, df_test_y, both_dfs=False):
    print("Predictions")
    print(
        pos_score + " " + neg_score + " " + neu_score + " " + comp_score + " " + like_count + " " + comment_count + " prediction" + " true value")

    df_test_y = df_test_y.reset_index(drop=True)

    if both_dfs:
        df_test_x = df_test_x.reset_index(drop=True)

    for i in randomlist:
        if both_dfs:
            print(df_test_x.iloc[i,:].to_string(header=False, index=False).replace("\n", " ") + " " + str(pred[i]) + " " + str(df_test_y.loc[[i]].to_string(header=False, index=False)))
        else:
            print(str(df_test_x[i]) + " " + str(pred[i]) + " " + str(df_test_y.loc[[i]].to_string(header=False, index=False)))

def train_random_forrest_regressor(file):
    df_test_x, df_test_y, df_training_x, df_training_y = prepare_data(file)

    rf = RandomForestRegressor(n_estimators=500, max_depth=100).fit(df_training_x, df_training_y)

    evaluate_model(rf, df_test_x, df_test_y, df_training_x, df_training_y)


def train_random_ada_boost_regressor(file):
    df_test_x, df_test_y, df_training_x, df_training_y = prepare_data(file)

    adaBoost = AdaBoostRegressor().fit(df_training_x, df_training_y)

    evaluate_model(adaBoost, df_test_x, df_test_y, df_training_x, df_training_y)


def evaluate_model(model, df_test_x, df_test_y, df_training_x, df_training_y):
    prediction = model.predict(df_test_x)

    mse = mean_squared_log_error(df_test_y, prediction)
    print("mean_squared_log_error: " + str(mse))

    print("overall score (r2 score): " + str(model.score(df_test_x, df_test_y)))

    print_predictions(prediction, df_test_x, df_test_y, True)


def prepare_data(file):
    df = get_training_data(file)

    df_sub_select = df[[pos_score, neg_score, neu_score, comp_score, like_count, comment_count, view_count]]

    row_count = len(df_sub_select.index)
    training_row_count = int(row_count * training_split)

    df_training = df_sub_select[:training_row_count]
    df_test = df_sub_select.iloc[training_row_count + 1:row_count, :]

    df_training_x = df_training[[pos_score, neg_score, neu_score, comp_score, like_count, comment_count]]
    df_test_x = df_test[[pos_score, neg_score, neu_score, comp_score, like_count, comment_count]]

    df_training_y = df_training[[view_count]]
    df_test_y = df_test[[view_count]]

    return df_test_x, df_test_y, df_training_x, df_training_y
