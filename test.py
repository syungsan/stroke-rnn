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
        self.alpha = tf.Variable(tr.alpha)

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


def predict(output_model, lof_model, lof_scaler, X):

    metrics_probs = output_model.predict(X, batch_size=1)
    metrics_probs = metrics_probs.reshape((len(metrics_probs), -1))
    lof_scaler.transform(metrics_probs)

    lof_scores = lof_model.decision_function(metrics_probs)
    y_preds = lof_model.predict(metrics_probs)

    return y_preds, lof_scores


def main():

    filelist = glob.glob(os.path.join("test/final", "*"))
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    y_trues = []
    y_preds = []
    lof_scores = []
    test_count = 0
    results = [["target", "recognition", "rnn_probability", "lof_score", "prediction", "filename"]]

    folder_paths = glob.glob("test/final/*")
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

            feature_raws = [[float(x) for x in y] for y in strokes]

            feature_xs = []
            feature_ys = []
            feature_2ds = [feature_xs, feature_ys]

            for j, feature_raw in enumerate(feature_raws):
                if j % 2 == 0:
                    feature_xs.append(feature_raw)
                elif j % 2 != 0:
                    feature_ys.append(feature_raw)

            X = ft.get_feature(feature_2ds=feature_2ds)
            X = np.array(X)
            scaler.transform([X])

            time_series_max_length = ft.interval_division_number * num_of_strokes[index]
            X = np.reshape(X, (1, time_series_max_length, tr.feature_max_length), order="F")

            # softmaxによる確率分布
            probabilities = model.predict(X).flatten()

            # リストの要素中最大値のインデックスを取得
            index_of_max = probabilities.argmax()

            recog = tr.stroke_dictionaries[num_of_strokes[index]][index_of_max]
            rnn_prob = probabilities[index_of_max]

            _y_preds, _lof_scores = predict(output_model, lof_model, lof_scaler, X)

            if _y_preds[0] == -1:
                y_pred = 0
            else:
                y_pred = 1

            y_preds.append(y_pred)
            lof_scores.append(_lof_scores[0])

            result = [target, recog, "{:.2f}%".format(rnn_prob * 100.0), "{:.2f}".format(lof_scores[0]), y_pred, filename]
            results.append(result)
            print(result)

    cm = confusion_matrix(y_trues, y_preds)
    pe.plot_confusion_matrix(cm, "./graphs/confusion_matrix.png")
    print(cm)

    accuracy = accuracy_score(y_trues, y_preds) * 100.0
    results.append(["accuracy: {}%".format(accuracy)])

    precision = precision_score(y_trues, y_preds)
    results.append(["precision: {}".format(precision)])

    recall = recall_score(y_trues, y_preds)
    results.append(["recall: {}".format(recall)])

    f1 = f1_score(y_trues, y_preds)
    results.append(["f1_score: {}".format(f1)])

    print("\nAccuracy: {}%".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1Score: {}".format(f1))

    roc_auc = pe.plot_roc_curve(y_trues, lof_scores, "./graphs/roc_curve.png")
    pr_auc = pe.plot_pr_curve(y_trues, lof_scores, "./graphs/pr_curve.png")

    results.append(["ROC-AUC: {}".format(roc_auc)])
    results.append(["PR-AUC: {}".format(pr_auc)])

    ft.write_csv("test/final/result.csv", results)
    print("\nall process completed...")


if __name__ == '__main__':

    # キーボードの入力待ち
    answer = input("テストを実施しますか？ 古い結果は上書きされます。 (Y/n)\n")

    if answer == "Y" or answer == "y" or answer == "":
        main()
    else:
        exit()
