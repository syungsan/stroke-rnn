#!/usr/bin/env python
# coding: utf-8

import os
from keras.models import load_model
import numpy as np
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import joblib
import glob
import feature as ft
import train as tr


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(5.) # 30.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha

def load_models(num_of_stroke):

    if os.path.exists("./models/{}/model.h5".format(num_of_stroke)):
        model = load_model("./models/{}/model.h5".format(num_of_stroke), custom_objects={"L2ConstrainLayer": L2ConstrainLayer})
        output_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    if os.path.exists("./models/{}/scaler.joblib".format(num_of_stroke)):
        scaler = joblib.load("./models/{}/scaler.joblib".format(num_of_stroke))

    if os.path.exists("./models/{}/lof_scaler.joblib".format(num_of_stroke)):
        lof_scaler = joblib.load("./models/{}/lof_scaler.joblib".format(num_of_stroke))

    if os.path.exists("./models/{}/lof_model.joblib".format(num_of_stroke)):
        lof_model = joblib.load("./models/{}/lof_model.joblib".format(num_of_stroke))

    return model, output_model, scaler, lof_model, lof_scaler

def main():

    correct = 0
    length_of_data = 0

    folder_paths = glob.glob("./test/final/*")
    if os.name == "nt":
        escape = "\\"
    else:
        escape = "/"
    num_of_strokes = [int(os.path.basename(d.rstrip(escape))) for d in folder_paths]

    for index, folder_path in enumerate(folder_paths):
        data_paths = glob.glob(folder_path + "/*")
        length_of_data += len(data_paths)

        for data_path in data_paths:
            filename = os.path.basename(data_path)
            target = filename[0]

            model, output_model, scaler, lof_model, lof_scaler = load_models(num_of_strokes[index])
            strokes = ft.read_csv(file_path=data_path, delimiter=",")

            X_pred = ft.get_feature(feature_raws=strokes)
            X_pred = np.array(X_pred)
            scaler.transform([X_pred])

            time_series_max_length = ft.SECTION_DIVISION_NUMBER * num_of_strokes[index]
            X_pred = np.reshape(X_pred, (1, time_series_max_length, tr.FEATURE_MAX_LENGTH), order="F")

            # softmaxによる確率分布
            probabilities = model.predict(X_pred).flatten()

            # リストの要素中最大値のインデックスを取得
            index_of_max = probabilities.argmax()

            recog = tr.STROKE_DICTIONARIES[num_of_strokes[index]][index_of_max]
            prob = "{:.2f}%".format(probabilities[index_of_max] * 100.0)

            lof = output_model.predict(X_pred)
            lof = lof.reshape((len(lof), -1))
            lof_scaler.transform(lof)

            is_error = lof_model.predict(lof)[0]
            score = "{:.2f}".format(lof_model.score_samples(lof)[0])
            print([recog, prob, is_error, score])

            if is_error == 1:
                if recog == target:
                    correct += 1
            else:
                if target == "不":
                    correct += 1

    accuracy = correct / length_of_data * 100.0
    print("\nFinal accuracy is " + accuracy + "%")

    ft.write_csv("./test/final/final_accuracy.csv", [[accuracy]])
    print("\nall process completed...")


if __name__ == '__main__':
    main()
