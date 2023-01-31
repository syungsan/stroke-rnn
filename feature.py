import codecs
import csv
import glob
from statistics import mean
import numpy as np
import os

SECTION_DIVISION_NUMBER = 5

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

def main():

    folder_paths = glob.glob("data/*/")
    if os.name == "nt":
        escape = "\\"
    else:
        escape = "/"
    num_of_strokes = [int(os.path.basename(d.rstrip(escape))) for d in folder_paths]

    for index, folder_path in enumerate(folder_paths):
        csv_paths = glob.glob(folder_path + "*.csv")

        features = []
        for csv_path in csv_paths:

            filename = os.path.basename(csv_path)
            target = filename[0]

            strokes = read_csv(file_path=csv_path, delimiter=",")

            if len(strokes) != num_of_strokes[index] * 2:
                continue

            feature = get_feature(feature_raws=strokes)
            feature.insert(0, target)

            features.append(feature)

        if len(features) != 0:
            write_csv("./data/train_{}.csv".format(num_of_strokes[index]), features)

    print("\nGet feature process was completed...")


if __name__ == "__main__":
    main()
