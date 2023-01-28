import os
from keras.models import load_model
from keras.models import Model
import joblib
import feature as ft
import numpy as np
import train as tr
import tensorflow as tf
import keras.backend as K


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(5.) # 30

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


def main():

    strokes = ft.read_csv(file_path="./test/test.csv", delimiter=",")
    num_of_stroke = int(len(strokes) / 2)

    if os.path.exists("./models/{}/model.h5".format(num_of_stroke)):
        model = load_model("./models/{}/model.h5".format(num_of_stroke), custom_objects={"L2ConstrainLayer": L2ConstrainLayer})
        output_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    if os.path.exists("./models/{}/scaler.joblib".format(num_of_stroke)):
        scaler = joblib.load("./models/{}/scaler.joblib".format(num_of_stroke))

    if os.path.exists("./models/{}/lof_scaler.joblib".format(num_of_stroke)):
        lof_scaler = joblib.load("./models/{}/lof_scaler.joblib".format(num_of_stroke))

    if os.path.exists("./models/{}/lof_model.joblib".format(num_of_stroke)):
        lof_model = joblib.load("./models/{}/lof_model.joblib".format(num_of_stroke))

    X_pred = ft.get_feature(feature_raws=strokes)
    X_pred = np.array(X_pred)
    scaler.transform([X_pred])

    time_series_max_length = ft.SECTION_DIVISION_NUMBER * num_of_stroke
    X_pred = np.reshape(X_pred, (1, time_series_max_length, tr.FEATURE_MAX_LENGTH), order="F")
    print(X_pred.shape)

    # softmaxによる確率分布
    probabilities = model.predict(X_pred).flatten()
    print(probabilities)

    # リストの要素中最大値のインデックスを取得
    index_of_max = probabilities.argmax()
    recog = tr.STROKE_DICTIONARIES[num_of_stroke][index_of_max]
    prob = probabilities[index_of_max]

    lof = output_model.predict(X_pred)
    lof = lof.reshape((len(lof), -1))
    lof_scaler.transform(lof)

    is_errors = lof_model.predict(lof)
    scores = lof_model.score_samples(lof)

    print([recog, prob, is_errors[0], scores[0]])


if __name__ == "__main__":
    main()
