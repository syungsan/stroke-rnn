#!/usr/bin/env python
# coding: utf-8

import sys
import os
import csv
from keras.models import load_model
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QImage, QPen, qRgb, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QSize
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import joblib
import fisher_yates_shuffle
from statistics import mean

STROKE_DICTIONARIES = {6: ["汚", "共", "会"], 7: ["求", "初", "芸"], 8: ["雨", "京", "到"]}
SECTION_DIVISION_NUMBER = 5
FEATURE_MAX_LENGTH = 3 * 2

def resource_path(relative):

    if hasattr(sys, "_MEIPASS"):
        # print("sys._MEIPASS:", sys._MEIPASS)
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)

def write_csv(file_path, list):

    try:
        # 書き込み UTF-8
        with open(file_path, "w", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerows(list)

    # 起こりそうな例外をキャッチ
    except FileNotFoundError as e:
        print(e)
    except csv.Error as e:
        print(e)


class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(5.) # 30.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha

def flatten_with_any_depth(nested_list):

    """深さ優先探索の要領で入れ子のリストをフラットにする関数"""
    # フラットなリストとフリンジを用意
    flat_list = []
    fringe = [nested_list]

    while len(fringe) > 0:
        node = fringe.pop(0)

        # ノードがリストであれば子要素をフリンジに追加
        # リストでなければそのままフラットリストに追加
        if isinstance(node, list):
            fringe = node + fringe
        else:
            flat_list.append(node)

    return flat_list

def section_average(list2ds, division_number):

    section_averages = []

    for list1ds in list2ds:
        if len(list1ds) < division_number:

            print("Extend list length with 0 padding...")
            additions = [0.0] * (division_number - len(list1ds))
            list1ds += additions

        size = int(len(list1ds) // division_number)
        mod = int(len(list1ds) % division_number)

        index_list = [size] * division_number
        if mod != 0:
            for i in range(mod):
                index_list[i] += 1

        averages = []
        i = 0

        for index in index_list:
            averages.append(mean(list1ds[i: i + index]))
            i += index

        section_averages.append(averages)

    return section_averages

def get_feature(feature_raws):

    feature_raws = [[float(x) for x in y] for y in feature_raws]

    feature_xs = []
    feature_ys = []
    features = [feature_xs, feature_ys]

    for index, feature_raw in enumerate(feature_raws):
        if index % 2 == 0:
            feature_xs.append(feature_raw)
        elif index % 2 != 0:
            feature_ys.append(feature_raw)

    all_features = []
    for feature in features:

        section_averages = section_average(list2ds=feature, division_number=SECTION_DIVISION_NUMBER)
        section_averages = np.array(section_averages)

        deltas = np.diff(section_averages, axis=1)
        deltas = np.pad(deltas, [(0, 0), (1, 0)], "constant")

        ddeltas = np.diff(deltas, axis=1)
        ddeltas = np.pad(ddeltas, [(0, 0), (1, 0)], "constant")

        sas = flatten_with_any_depth(nested_list=section_averages.tolist())
        ds = flatten_with_any_depth(nested_list=deltas.tolist())
        dds = flatten_with_any_depth(nested_list=ddeltas.tolist())

        all_features.append(sas + ds + dds)

    return flatten_with_any_depth(nested_list=all_features)


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        # self.setWindowIcon(QIcon(resource_path("./icons/simple_icon.ico")))
        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("Stroke RNN")

        self.reference_title_label = QLabel()
        self.reference_label = QLabel()
        self.reference_num_of_stroke_label = QLabel()
        self.num_of_write_stroke_label = QLabel()
        self.dont_mutch_label = QLabel()
        self.recognition_label = QLabel()
        self.probability_label = QLabel()

        reference_title_font = QFont()
        reference_title_font.setPointSize(18)
        self.reference_title_label = QLabel("Reference")
        self.reference_title_label.setFont(reference_title_font)
        self.reference_title_label.setAlignment(Qt.AlignCenter)

        reference_font = QFont()
        reference_font.setPointSize(40)
        self.reference_label = QLabel("")
        self.reference_label.setFont(reference_font)
        self.reference_label.setAlignment(Qt.AlignCenter)

        reference_num_of_stroke_font = QFont()
        reference_num_of_stroke_font.setPointSize(18)
        self.reference_num_of_stroke_label = QLabel("Num of stroke: 0")
        self.reference_num_of_stroke_label.setFont(reference_num_of_stroke_font)
        self.reference_num_of_stroke_label.setAlignment(Qt.AlignCenter)

        num_of_write_stroke_font = QFont()
        num_of_write_stroke_font.setPointSize(18)
        self.num_of_write_stroke_label = QLabel("Write stroke: 0")
        self.num_of_write_stroke_label.setFont(num_of_write_stroke_font)
        self.num_of_write_stroke_label.setAlignment(Qt.AlignCenter)

        correctness_font = QFont()
        correctness_font.setPointSize(12)
        self.correctness_label = QLabel("")
        self.correctness_label.setFont(correctness_font)
        self.correctness_label.setAlignment(Qt.AlignCenter)

        recognition_font = QFont()
        recognition_font.setPointSize(70)
        self.recognition_label = QLabel("")
        self.recognition_label.setFont(recognition_font)
        self.recognition_label.setAlignment(Qt.AlignCenter)

        probability_font = QFont()
        probability_font.setPointSize(30)
        self.probability_label = QLabel("")
        self.probability_label.setFont(probability_font)
        self.probability_label.setAlignment(Qt.AlignCenter)

        button_font = QFont()
        button_font.setPointSize(18)

        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next)
        next_button.setFixedSize(200, 50)
        next_button.setFont(button_font)

        recog_button = QPushButton("Recognition")
        recog_button.clicked.connect(self.recognition)
        recog_button.setFixedSize(200, 50)
        recog_button.setFont(button_font)

        v_box = QVBoxLayout()
        v_box.addWidget(self.reference_title_label)
        v_box.addWidget(self.reference_label)
        v_box.addWidget(self.reference_num_of_stroke_label)
        v_box.addWidget(self.num_of_write_stroke_label)
        v_box.addWidget(self.correctness_label)
        v_box.addWidget(self.recognition_label)
        v_box.addWidget(self.probability_label)
        v_box.addWidget(next_button)
        v_box.addWidget(recog_button)
        v_box.addSpacing(20)

        h_box = QHBoxLayout()
        self.canvas = Canvas(parent=self)
        self.canvas.setFixedSize(600, 600)
        h_box.addWidget(self.canvas)
        h_box.addLayout(v_box)

        main_window = QWidget()
        main_window.setLayout(h_box)
        self.setCentralWidget(main_window)
        self.setFixedSize(900, 640)
        self.set_question()

        self.show()

    def set_question(self):

        self.num_of_stroke = fisher_yates_shuffle.fisher_yates_shuffle(list(STROKE_DICTIONARIES.keys()))[0]
        reference_word = fisher_yates_shuffle.fisher_yates_shuffle(STROKE_DICTIONARIES[self.num_of_stroke])[0]
        self.reference_label.setText(reference_word)
        self.reference_num_of_stroke_label.setText("Num of stroke: {}".format(self.num_of_stroke))

        self.load_models()

    def load_models(self):

        self.model = None
        self.output_model = None
        self.scaler = None
        self.lof_scaler = None
        self.lof_model = None

        if os.path.exists("./models/{}/model.h5".format(self.num_of_stroke)):
            self.model = load_model("./models/{}/model.h5".format(self.num_of_stroke), custom_objects={"L2ConstrainLayer": L2ConstrainLayer})
            self.output_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

        if os.path.exists("./models/{}/scaler.joblib".format(self.num_of_stroke)):
            self.scaler = joblib.load("./models/{}/scaler.joblib".format(self.num_of_stroke))

        if os.path.exists("./models/{}/lof_scaler.joblib".format(self.num_of_stroke)):
            self.lof_scaler = joblib.load("./models/{}/lof_scaler.joblib".format(self.num_of_stroke))

        if os.path.exists("./models/{}/lof_model.joblib".format(self.num_of_stroke)):
            self.lof_model = joblib.load("./models/{}/lof_model.joblib".format(self.num_of_stroke))

    def next(self):

        self.canvas.resetImage()
        self.correctness_label.setText("")
        self.recognition_label.setText("")
        self.probability_label.setText("")
        self.set_question()

    def recognition(self):

        self.canvas.save_image("./test/test.png")

        strokes = self.canvas.strokes
        num_of_write_stroke = self.canvas.num_of_stroke

        if num_of_write_stroke != self.num_of_stroke:
            self.correctness_label.setText("Number of Stroke\ndon't match.")
            return

        X_pred = get_feature(feature_raws=strokes)

        X_pred = np.array(X_pred)
        self.scaler.transform([X_pred])

        time_series_max_length = SECTION_DIVISION_NUMBER * num_of_write_stroke
        X_pred = np.reshape(X_pred, (1, time_series_max_length, FEATURE_MAX_LENGTH), order="F")
        print(X_pred.shape)

        # softmaxによる確率分布
        probabilities = self.model.predict(X_pred).flatten()
        print(probabilities)

        # リストの要素中最大値のインデックスを取得
        index_of_max = probabilities.argmax()
        recog = STROKE_DICTIONARIES[self.num_of_stroke][index_of_max]
        prob = "{:.2f}%".format(probabilities[index_of_max] * 100.0)

        lof = self.output_model.predict(X_pred)
        lof = lof.reshape((len(lof), -1))
        self.lof_scaler.transform(lof)

        is_error = self.lof_model.predict(lof)[0]
        score = "{:.2f}".format(self.lof_model.score_samples(lof)[0])
        print([recog, prob, is_error, score])

        if is_error == 1:
            self.recognition_label.setText(recog)
            self.probability_label.setText(prob)
        else:
            self.recognition_label.setText("Error")
            self.probability_label.setText(score)


class Canvas(QWidget):

    def __init__(self, parent=None):
        super(Canvas, self).__init__(parent)

        self.myPenWidth = 40
        self.myPenColor = Qt.black
        self.image = QImage()
        self.check = False
        self.stroke_x = []
        self.stroke_y = []
        self.strokes = []
        self.num_of_stroke = 0
        self.parent = self.parent()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPos = event.pos()
            self.check = True

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.check:
            self.drawLine(event.pos())
            self.stroke_x.append(event.x())
            self.stroke_y.append(event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.check:
            self.drawLine(event.pos())
            self.strokes.append(self.stroke_x)
            self.strokes.append(self.stroke_y)
            self.stroke_x = []
            self.stroke_y = []
            self.num_of_stroke += 1
            self.parent.num_of_write_stroke_label.setText("Write stroke: {}".format(self.num_of_stroke))
            self.check = False

    def drawLine(self, endPos):
        painter = QPainter(self.image)
        # painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(
            QPen(self.myPenColor, self.myPenWidth, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        )
        painter.drawLine(self.lastPos, endPos)
        self.update()
        self.lastPos = QPoint(endPos)

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = event.rect()
        painter.drawImage(rect, self.image, rect)

    def resizeEvent(self, event):
        if self.image.width() < self.width() or self.image.height() < self.height():
            changeWidth = max(self.width(), self.image.width())
            changeHeight = max(self.height(), self.image.height())
            self.image = self.resizeImage(self.image, QSize(changeWidth, changeHeight))
            self.update()

    def resizeImage(self, image, newSize):
        changeImage = QImage(newSize, QImage.Format_RGB32)
        changeImage.fill(qRgb(255, 255, 255))
        painter = QPainter(changeImage)
        painter.drawImage(QPoint(0, 0), image)
        return changeImage

    def save_image(self, filename):
        write_csv("./test/test.csv", self.strokes)
        if self.image.save(filename):
            return True
        else:
            return False

    def resetImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.update()
        self.strokes = []
        self.num_of_stroke = 0
        self.parent.num_of_write_stroke_label.setText("Write stroke: {}".format(0))

def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()