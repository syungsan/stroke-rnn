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
import plot_evaluation as pe
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


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

    filelist = glob.glob(os.path.join("./test/final", "*"))
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    y_trues = []
    y_preds = []
    y_scores = []
    test_count = 0
    results = [["target", "recognition", "rnn_probability", "is_error", "lof_score", "prediction", "total_score"]]

    folder_paths = glob.glob("./test/final/*")
    if os.name == "nt":
        escape = "\\"
    else:
        escape = "/"
    num_of_strokes = [int(os.path.basename(d.rstrip(escape))) for d in folder_paths]

    for index, folder_path in enumerate(folder_paths):
        data_paths = glob.glob(folder_path + "/*")

        for data_path in data_paths:
            filename = os.path.basename(data_path)
            target = filename[0]

            if target == "不":
                y_trues.append(0)
            else:
                y_trues.append(1)

            test_count += 1
            print("\nTest count: {}".format(test_count))

            model, output_model, scaler, lof_model, lof_scaler = load_models(num_of_strokes[index])
            strokes = ft.read_csv(file_path=data_path, delimiter=",")

            X = ft.get_feature(feature_raws=strokes)
            X = np.array(X)
            scaler.transform([X])

            time_series_max_length = ft.SECTION_DIVISION_NUMBER * num_of_strokes[index]
            X = np.reshape(X, (1, time_series_max_length, tr.FEATURE_MAX_LENGTH), order="F")

            # softmaxによる確率分布
            probabilities = model.predict(X).flatten()

            # リストの要素中最大値のインデックスを取得
            index_of_max = probabilities.argmax()

            recog = tr.STROKE_DICTIONARIES[num_of_strokes[index]][index_of_max]
            rnn_prob = probabilities[index_of_max]

            lof = output_model.predict(X)
            lof = lof.reshape((len(lof), -1))
            lof_scaler.transform(lof)

            error = lof_model.predict(lof)[0]
            lof_score = lof_model.score_samples(lof)[0]

            if error == 1:
                is_error = False

                if recog == target:
                    y_pred = 1
                else:
                    y_pred = 0
            else:
                is_error = True
                y_pred = 0

            y_preds.append(y_pred)

            if y_pred == 0:
                y_rnn_prob = ((1.0 - rnn_prob) * 0.5)
            else:
                y_rnn_prob = rnn_prob

            y_lof_score = 1.0 / (abs(lof_score) + 0.03) # 0.03はあてずっぽう補正（0.1-0.07）0.07はlofのcontamination
            y_score = (y_rnn_prob + y_lof_score) * 0.5
            y_scores.append(y_score)

            result = [target, recog, "{:.2f}%".format(rnn_prob * 100.0), is_error, "{:.2f}".format(lof_score), y_pred, y_score]
            results.append(result)
            print(result)

    cm = confusion_matrix(y_trues, y_preds)
    pe.plot_confusion_matrix(cm)
    print(cm)

    final_accuracy = accuracy_score(y_trues, y_preds) * 100.0
    results.append(["accuracy: {}%".format(final_accuracy)])

    precision = precision_score(y_trues, y_preds)
    results.append(["precision: {}".format(precision)])

    recall = recall_score(y_trues, y_preds)
    results.append(["recall: {}".format(recall)])

    f1 = f1_score(y_trues, y_preds)
    results.append(["f1_score: {}".format(f1)])

    print("\nAccuracy: {}%".format(final_accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1Score: {}".format(f1))

    roc_auc = pe.plot_roc_curve(y_trues, y_scores)
    pr_auc = pe.plot_pr_curve(y_trues, y_scores)

    results.append(["ROC-AUC: {}".format(roc_auc)])
    results.append(["PR-AUC: {}".format(pr_auc)])

    ft.write_csv("./test/final/result.csv", results)
    print("\nall process completed...")


if __name__ == '__main__':

    # キーボードの入力待ち
    answer = input("テストを実施しますか？ 古い結果は上書きされます。 (Y/n)\n")

    if answer == "Y" or answer == "y" or answer == "":
        main()
    else:
        exit()
