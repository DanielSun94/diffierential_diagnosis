import random
import numpy as np
import os
import pickle
from data_preprocess.data_reformat import load_data, reconstruct_data
import hanlp
from config import topic_model_first_emr_parse_list, topic_model_admission_parse_list, diagnosis_map, \
    word_count_path, skip_word_set


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



def bag_of_word_reorganize(vocab_size):
    admission_data_dict, first_emr_record_dict = load_data(read_from_cache=False)
    data = reconstruct_data(admission_data_dict, first_emr_record_dict, topic_model_admission_parse_list,
                            topic_model_first_emr_parse_list)
    word_index_map, reformat_data = word_index_convert(data, word_count_path, vocab_size)
    return word_index_map, reformat_data


def word_index_convert(data, save_path, vocab_size, read_from_cache=True):
    if read_from_cache and os.path.exists(word_count_path):
        reformat_data = pickle.load(open(save_path, 'rb'))
    else:
        HanLP = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)
        data_list = []
        for key in data:
            data_list.append([key, data[key][0], diagnosis_map[data[key][1]]])
        content_list = [item[1] for item in data_list]
        tokenize_data = HanLP(content_list)
        reformat_data = list()
        for i in range(len(tokenize_data)):
            reformat_data.append([data_list[i][0], tokenize_data[i], data_list[i][2]])
        pickle.dump(reformat_data, open(save_path, 'wb'))

    # len_count = [len(item[1]) for item in reformat_data]
    # print('average token length: {}'.format(np.average(len_count)))
    # print('max token length: {}'.format(np.max(len_count)))
    # print('min token length: {}'.format(np.min(len_count)))

    word_count_map = dict()
    for item in reformat_data:
        for word in item[1]:
            if word in skip_word_set:
                continue
            if word not in word_count_map:
                word_count_map[word] = 0
            word_count_map[word] += 1

    word_count_list = list()
    for key in word_count_map:
        word_count_list.append([key, word_count_map[key]])
    word_count_list = sorted(word_count_list, key=lambda x: x[1], reverse=True)
    valid_word = [word_count_list[i][0] for i in range(vocab_size-1)]

    word_index_map = dict()
    for item in valid_word:
        word_index_map[item] = len(word_index_map)

    tokenize_data = dict()
    for item in reformat_data:
        key, content, diagnosis = item
        content_idx_list = np.zeros(vocab_size)
        for word in content:
            if word in word_index_map:
                content_idx_list[word_index_map[word]] += 1
            else:
                content_idx_list[vocab_size-1] += 1
        tokenize_data[key] = content_idx_list, diagnosis
    return word_index_map, tokenize_data