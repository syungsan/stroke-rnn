import keras
import numpy as np
import matplotlib.pyplot as plt
import feature as ft
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.neighbors import LocalOutlierFactor
import joblib
import keras.backend as K
import tensorflow as tf
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
import shutil
from keras.callbacks import CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.manifold import TSNE
import glob

STROKE_DICTIONARIES = {6: ["汚", "共", "会"], 7: ["求", "初", "芸"], 8: ["雨", "京", "到"]}
FEATURE_MAX_LENGTH = 3 * 2 # 微分成分含めた(3) × xy成分(2)


# L2-constrained Softmax Loss
class L2ConstrainLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(L2ConstrainLayer, self).__init__(**kwargs)
        self.alpha = tf.Variable(5.) # 30

    def call(self, inputs):
        return K.l2_normalize(inputs, axis=1) * self.alpha


def plot_result(history, num_of_stroke):

    """
    plot result
    全ての学習が終了した後に、historyを参照して、accuracyとlossをそれぞれplotする
    """

    # accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="acc", marker=".")
    plt.plot(history.history["val_accuracy"], label="val_acc", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Accuracy")
    plt.savefig("./graphs/{}/accuracy.png".format(num_of_stroke))
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history["loss"], label="loss", marker=".")
    plt.plot(history.history["val_loss"], label="val_loss", marker=".")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(loc="best")
    plt.title("Loss")
    plt.savefig("./graphs/{}/loss.png".format(num_of_stroke))
    plt.show()

# データの分布の様子を次元を落として表示
def plot_tsene(X, model, output_model, num_of_stroke):

    color_codes = ["red", "blue", "green", "black", "magenta", "cyan", "grey", "aqua", "springgreen", "salmon"]
    test_range = range(int(len(X)))

    output = model.predict(X)
    markers = []
    colors = []

    for i in test_range:
        markers.append(np.argmax(output[i, :]))
        colors.append(color_codes[np.argmax(output[i, :])])

    hidden_output = output_model.predict(X)
    X_reduced = TSNE(n_components=2, random_state=0).fit_transform(hidden_output)

    plt.figure(figsize=(10, 10))
    for i in test_range:
        plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c=colors[i], marker="${}$".format(markers[i]), s=40)

    plt.savefig("./graphs/{}/t-SNE.png".format(num_of_stroke))
    plt.show()

def get_train(train2ds, num_of_stroke):

    Xs = []
    ys = []

    for train1ds in train2ds:
        Xs.append(train1ds[1:])
        ys.append(STROKE_DICTIONARIES[num_of_stroke].index(train1ds[0]))

    X = [[float(x) for x in y] for y in Xs]

    return np.array(X), ys

# 異常検出モデルの作成
def lof(output_model, X_train, num_of_stroke):

    X_train = output_model.predict(X_train)
    X_train = X_train.reshape((len(X_train), -1))

    lof_scaler = MinMaxScaler()
    lof_scaler.fit(X_train)
    lof_scaler.transform(X_train)

    print("anomaly detection model creating...")
    # contamination = 学習データにおける外れ値の割合（大きいほど厳しく小さいほど緩い）
    # example-> k(n_neighbors=10**0.5=3) 10=num of class
    model = LocalOutlierFactor(n_neighbors=3, novelty=True, contamination=0.07) # 20, novelty=True, contamination=0.001)
    model.fit(X_train)

    joblib.dump(lof_scaler, "./models/{}/lof_scaler.joblib".format(num_of_stroke))
    joblib.dump(model, "./models/{}/lof_model.joblib".format(num_of_stroke), compress=True)

def main(epochs=5, batch_size=128):

    if os.path.isdir("./models"):
        shutil.rmtree("./models")
    os.mkdir("./models")

    if os.path.isdir("./graphs"):
        shutil.rmtree("./graphs")
    os.mkdir("./graphs")

    file_paths = glob.glob("./data/*.csv")
    num_of_strokes = [int(os.path.splitext(os.path.basename(f))[0][-1]) for f in file_paths]

    for index in range(len(file_paths)):

        strokes = ft.read_csv(file_path="./data/train_{}.csv".format(num_of_strokes[index]), delimiter=",")
        X, y = get_train(train2ds=strokes, num_of_stroke=num_of_strokes[index])

        # one-hot vector形式に変換する
        num_of_category = len(STROKE_DICTIONARIES[num_of_strokes[index]])
        y = to_categorical(y, num_of_category)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        scaler = MinMaxScaler()
        scaler.fit(X_train)
        scaler.transform(X_train)
        scaler.transform(X_test)

        time_series_max_length = ft.SECTION_DIVISION_NUMBER * num_of_strokes[index]

        X_train = np.reshape(X_train, (X_train.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")
        X_test = np.reshape(X_test, (X_test.shape[0], time_series_max_length, FEATURE_MAX_LENGTH), order="F")

        #Initializing model
        model = keras.models.Sequential()

        #Adding the model layers
        model.add(keras.layers.LSTM(256, input_shape=(X_train.shape[1:]), return_sequences=True))
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.LSTM(128))
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.2))
        model.add(L2ConstrainLayer())
        model.add(keras.layers.Dense(num_of_category, activation='softmax'))

        #Compiling the model
        model.compile(loss="categorical_crossentropy",
                      optimizer=Adam(learning_rate=0.0001, amsgrad=True), # RMSprop(), # (learning_rate=1e-4),
                      metrics=["accuracy"])

        model.summary()

        os.mkdir("./models/{}".format(num_of_strokes[index]))

        # callback function
        csv_cb = CSVLogger("./models/{}/train_log.csv".format(num_of_strokes[index]))
        fpath = "./models/" + str(num_of_strokes[index]) + "/model-{epoch:02d}-{loss:.2f}-{accuracy:.2f}-{val_loss:.2f}-{val_accuracy:.2f}-.h5"
        cp_cb = ModelCheckpoint(filepath=fpath, monitor="val_loss", verbose=1, save_best_only=True, mode="auto")
        es_cb = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="auto")
        tb_cb = TensorBoard(log_dir="./tensor_log/{}".format(num_of_strokes[index]), histogram_freq=1)

        #Fitting data to the model
        history = model.fit(
            x=X_train, y=y_train,
            steps_per_epoch=X_train.shape[0] // batch_size,
            epochs=epochs,
            validation_split=0.1,
            callbacks=[csv_cb, cp_cb, es_cb, tb_cb],
            verbose=1)

        # result
        score = model.evaluate(X_test, y_test, verbose=0)
        print("Test loss of {} stroke: {}".format(num_of_strokes[index], score[0]))
        print("Test accuracy of {} stroke: {}".format(num_of_strokes[index], score[1]))

        recog_results = [["accuracy", "loss"], [score[1], score[0]]]
        ft.write_csv("./models/{}/recog_result.csv".format(num_of_strokes[index]), recog_results)

        os.mkdir("./graphs/{}".format(num_of_strokes[index]))

        plot_result(history, num_of_stroke=num_of_strokes[index])

        model.save("./models/{}/model.h5".format(num_of_strokes[index]))
        joblib.dump(scaler, "./models/{}/scaler.joblib".format(num_of_strokes[index]))

        output_model = Model(inputs=model.input, outputs=model.layers[-2].output)
        plot_tsene(X=X_test, model=model, output_model=output_model, num_of_stroke=num_of_strokes[index])
        lof(output_model=output_model, X_train=X_train, num_of_stroke=num_of_strokes[index])

        K.clear_session()

    print("\nall process completed...")


if __name__ == "__main__":

    # キーボードの入力待ち
    answer = input("トレーニングを実施しますか？ 古いモデルは上書きされます。 (Y/n)\n")

    if answer == "Y" or answer == "y" or answer == "":
        epochs = 50
        batch_size = 128
        main(epochs, batch_size)
    else:
        exit()
