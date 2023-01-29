#!/usr/bin/env python
# coding: utf-8

import sys
import os
from keras.models import load_model
# from keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import img_to_array, load_img
import cv2
import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QImage, QPen, qRgb, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint, QSize
import tensorflow as tf
from keras.models import Model
import keras.backend as K
import joblib


def resource_path(relative):

    if hasattr(sys, "_MEIPASS"):
        # print("sys._MEIPASS:", sys._MEIPASS)
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(relative)

class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(5.) # 30.

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowIcon(QIcon(resource_path("./icon/simple_icon.ico")))

        self.number_labels = list(range(10))

        if os.path.exists(resource_path("./model/model.h5")):
            self.model = load_model(resource_path("./model/model.h5"), custom_objects={"L2ConstrainLayer": L2ConstrainLayer})
            self.output_model = Model(inputs=self.model.input, outputs=self.model.layers[-2].output)

        if os.path.exists(resource_path("./model/scaler.joblib")):
            self.scaler = joblib.load(resource_path("./model/scaler.joblib"))

        if os.path.exists(resource_path("./model/lof_model.joblib")):
            self.lof_model = joblib.load(resource_path("./model/lof_model.joblib"))

        self.canvas = Canvas()
        self.recognition_label = QLabel()
        self.probability_label = QLabel()

        self.initUI()

    def initUI(self):

        self.setWindowTitle("Number Recognition")

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

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear)
        clear_button.setFixedSize(200, 50)
        clear_button.setFont(button_font)

        recog_button = QPushButton("Recognition")
        recog_button.clicked.connect(self.recognition)
        recog_button.setFixedSize(200, 50)
        recog_button.setFont(button_font)

        v_box = QVBoxLayout()
        v_box.addWidget(self.recognition_label)
        v_box.addWidget(self.probability_label)
        v_box.addWidget(clear_button)
        v_box.addWidget(recog_button)
        v_box.addSpacing(20)

        h_box = QHBoxLayout()
        self.canvas.setFixedSize(600, 600)
        h_box.addWidget(self.canvas)
        h_box.addLayout(v_box)

        main_window = QWidget()
        main_window.setLayout(h_box)
        self.setCentralWidget(main_window)
        self.setFixedSize(900, 640)

        self.show()

    def clear(self):

        self.canvas.resetImage()
        self.recognition_label.setText("")
        self.probability_label.setText("")

    def recognition(self):

        self.canvas.save_image("./test.png")
        img = img_to_array(load_img("./test.png", target_size=(28, 28))).astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = img.reshape((1,) + img.shape + (1,)).astype(np.float32) / 255.0

        probabilitys = self.model.predict(img).flatten()
        index_of_max = probabilitys.argmax()
        number = str(self.number_labels[index_of_max])
        probability = "{:.2f}%".format(probabilitys[index_of_max] * 100)

        X = self.output_model.predict(img)
        X = X.reshape((len(X), -1))
        self.scaler.transform(X)

        is_error = self.lof_model.predict(X)[0]
        score = "{:.2f}".format(self.lof_model.score_samples(X)[0])

        if is_error == 1:
            self.recognition_label.setText(number)
            self.probability_label.setText(probability)
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.lastPos = event.pos()
            self.check = True

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.check:
            self.drawLine(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.check:
            self.drawLine(event.pos())
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
        if self.image.save(filename):
            return True
        else:
            return False

    def resetImage(self):
        self.image.fill(qRgb(255, 255, 255))
        self.update()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
