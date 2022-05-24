import random
from config import diagnosis_map
import numpy as np


def five_fold_datasets(data):
    """
    五折交叉验证
    """
    data_list = []
    for key in data:
        single_data, diagnosis = data[key][0], data[key][1]
        data_list.append([key, single_data, diagnosis])
    index_list = [i for i in range(len(data_list))]
    random.shuffle(index_list)

    shuffled_data_list = []
    for index in index_list:
        shuffled_data_list.append(data_list[index])

    fold_size = len(shuffled_data_list) // 5
    shuffled_data = [
        shuffled_data_list[0: fold_size],
        shuffled_data_list[fold_size: fold_size * 2],
        shuffled_data_list[fold_size * 2: fold_size * 3],
        shuffled_data_list[fold_size * 3: fold_size * 4],
        shuffled_data_list[fold_size * 4:],
    ]
    return shuffled_data


def dataset_format(dataset):
    feature, label = list(), list()
    for item in dataset:
        feature.append(item[1])
        label.append(item[2])
    return feature, label


def evaluation(predict, label):
    predict_label = np.argmax(predict, axis=1)
    accuracy = np.sum(label == predict_label) / len(label)
    return accuracy