import codecs
import csv
import glob
from statistics import mean
import numpy as np
import os
import fisher_yates_shuffle


# 1セクションの分割数
interval_division_number = 5
times_of_data_augmentation = 0


def read_csv(file_path, delimiter):

    lists = []
    file = codecs.open(file_path, "r", "utf-8")

    reader = csv.reader(file, delimiter=delimiter)

    for line in reader:
        lists.append(line)

    file.close
    return lists


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


def interval_division_average(list2ds, division_number):

    idas = []

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

        idas.append(averages)

    return idas


def get_feature(feature_2ds):

    all_features = []
    for feature in feature_2ds:

        idas = interval_division_average(list2ds=feature, division_number=interval_division_number)
        idas = np.array(idas)

        deltas = np.diff(idas, axis=1)
        deltas = np.pad(deltas, [(0, 0), (1, 0)], "constant")

        ddeltas = np.diff(deltas, axis=1)
        ddeltas = np.pad(ddeltas, [(0, 0), (1, 0)], "constant")

        _idas = flatten_with_any_depth(nested_list=idas.tolist())
        _deltas = flatten_with_any_depth(nested_list=deltas.tolist())
        _ddeltas = flatten_with_any_depth(nested_list=ddeltas.tolist())

        all_features.append(_idas + _deltas + _ddeltas)

    return flatten_with_any_depth(nested_list=all_features)


def main():

    folder_paths = glob.glob("data/*/")
    if os.name == "nt":
        escape = "\\"
    else:
        escape = "/"
    num_of_strokes = [int(os.path.basename(d.rstrip(escape))) for d in folder_paths]

    for i, folder_path in enumerate(folder_paths):
        csv_paths = glob.glob(folder_path + "*.csv")

        features = []
        for csv_path in csv_paths:

            filename = os.path.basename(csv_path)
            target = filename[0]

            strokes = read_csv(file_path=csv_path, delimiter=",")

            if len(strokes) != num_of_strokes[i] * 2:
                continue

            feature_raws = [[float(x) for x in y] for y in strokes]

            feature_xs = []
            feature_ys = []
            feature_2ds = [feature_xs, feature_ys]

            for j, feature_raw in enumerate(feature_raws):
                if j % 2 == 0:
                    feature_xs.append(feature_raw)
                elif j % 2 != 0:
                    feature_ys.append(feature_raw)

            feature = get_feature(feature_2ds=feature_2ds)
            feature.insert(0, target)
            features.append(feature)

            index_list = list(range(num_of_strokes[i]))

            for k in range(times_of_data_augmentation):
                index_list = fisher_yates_shuffle.fisher_yates_shuffle(index_list)

                xs = []
                ys = []

                for l in range(len(index_list)):
                    xs.append(feature_xs[index_list[l]])
                    ys.append(feature_ys[index_list[l]])

                feature_2ds = [xs, ys]

                feature = get_feature(feature_2ds=feature_2ds)
                feature.insert(0, target)
                features.append(feature)

        if len(features) != 0:
            write_csv("./data/train_{}.csv".format(num_of_strokes[i]), features)

    print("\nGet feature process was completed...")


if __name__ == "__main__":
    main()
