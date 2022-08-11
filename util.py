import math
import numpy as np
from config import classify_weight_csv, performance_tendency_csv, topic_word_distribution_csv, diagnosis_map, \
    device, representation_pkl, perplexity_csv
import pickle
import datetime
import csv
import torch
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from mimic_data_reformat import mimic_load_data
from hzsph_data_reformat import hzsph_load_data


def dataset_format(dataset):
    feature, label, representation, string = list(), list(), list(), list()
    for item in dataset:
        if len(item) == 3:
            feature.append(item[1])
            label.append(item[2])
        elif len(item) == 5:
            feature.append(item[1])
            representation.append(item[2])
            label.append(item[3])
            string.append(item[4])
        else:
            raise ValueError('')
    if len(dataset[0]) == 3:
        return feature, label
    elif len(dataset[0]) == 5:
        return feature, representation, label, string
    else:
        raise ValueError('')


def dataset_selection(dataset_name, vocab_size, diagnosis_size, cut_length, read_from_cache):
    if dataset_name == 'mimic-iii':
        five_fold_data, word_index_map = mimic_load_data(vocab_size, diagnosis_size, read_from_cache, cut_length)
    elif dataset_name == 'hzsph':
        five_fold_data, word_index_map = hzsph_load_data(read_from_cache, vocab_size, cut_length)
    else:
        raise ValueError('')
    return five_fold_data, word_index_map


def evaluation(predict_prob, label):
    predict = np.argmax(predict_prob, axis=1)
    micro_recall = recall_score(label, predict, average='micro')
    macro_recall = recall_score(label, predict, average='macro')
    micro_precision = precision_score(label, predict, average='micro')
    macro_precision = precision_score(label, predict, average='macro')
    micro_f1 = f1_score(label, predict, average='micro')
    macro_f1 = f1_score(label, predict, average='macro')
    accuracy = accuracy_score(label, predict)

    # micro_auc = roc_auc_score(label, predict_prob, multi_class='ovo')
    # macro_auc = roc_auc_score(label, predict_prob, multi_class='ovr')
    performance = {
        'accuracy': accuracy,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        # 'micro_auc': micro_auc,
        # 'macro_auc': macro_auc,
    }
    return performance


def write_model_info(classify_model, representation_model, word_index_map, cv_para):
    dataset_name = cv_para['dataset_name']
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")

    topic_word_weight = representation_model.ntm.topics.get_topics()
    classify_weight = classify_model.coefs_[0]
    index_word_list = [[word_index_map[key], key]for key in word_index_map]
    index_word_list = sorted(index_word_list, key=lambda x: x[0])
    # topic word distribution说明
    topic_dict = {}
    for i in range(len(topic_word_weight)):
        single_topic_word_distribution = topic_word_weight[i].detach().to('cpu').numpy()
        map_list = []
        # 此处-1是因为最后一个index惯例给了unknown
        # assert len(index_word_list) == len(single_topic_word_distribution)-1
        for j in range(len(index_word_list)):
            map_list.append([index_word_list[j][1], single_topic_word_distribution[j]])
        map_list = sorted(map_list, key=lambda x: x[1], reverse=True)
        topic_dict[i] = map_list
    topic_word_write = [['' for _ in range(len(topic_dict)*3+4)] for _ in range(len(topic_dict[0])+4)]
    for topic_no in range(len(topic_dict)):
        topic_word_write[0][topic_no*3] = 'topic:' + str(topic_no)
        for word_no in range(len(topic_dict[topic_no])):
            topic_word_write[word_no + 1][topic_no * 3] = topic_dict[topic_no][word_no][0]
            topic_word_write[word_no + 1][topic_no * 3 + 1] = topic_dict[topic_no][word_no][1]
    write_with_args = [[key, cv_para[key]] for key in cv_para]
    for line in topic_word_write:
        write_with_args.append(line)
    with open(topic_word_distribution_csv.format(dataset_name, timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_with_args)

    # 数据
    classify_weight_write = [['' for _ in range(len(classify_weight)+4)] for _ in range(len(classify_weight[0])+4)]
    for i in range(len(classify_weight)):
        classify_weight_write[0][i+1] = str(i)
    for key in diagnosis_map:
        classify_weight_write[diagnosis_map[key]+1][0] = key
    for i in range(len(classify_weight)):
        for j in range(len(classify_weight[i])):
            classify_weight_write[j+1][i+1] = classify_weight[i, j]
    write_with_args = [[key, cv_para[key]] for key in cv_para]
    for line in classify_weight_write:
        write_with_args.append(line)
    with open(classify_weight_csv.format(dataset_name, timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_with_args)


def write_tendency(tendency_list, cv_para):
    dataset_name = cv_para['dataset_name']
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    write_info = []
    for key in cv_para:
        write_info.append([key, cv_para[key]])
    write_info.append([])

    write_info.append(['epoch num', "metric", 'value'])
    for metric in 'accuracy', 'macro_recall', 'macro_precision', 'macro_f1', 'micro_recall', 'micro_precision', \
                  "micro_f1":
        for item in tendency_list:
            epoch_num, performance_dict = item
            performance = performance_dict[metric]
            write_info.append([epoch_num, metric, performance])
    with open(performance_tendency_csv.format(dataset_name, timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_info)


def write_perplexity(model, test_dataset, cv_para):
    dataset_name = cv_para['dataset_name']
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    test_feature = torch.FloatTensor(np.array(test_dataset[0])).to(device)
    test_representation = model.ntm.get_topic_distribution(test_feature, 'inference').detach().to(
        'cpu').numpy()
    topic_word_distribution = model.ntm.get_topics().detach().to('cpu').numpy()
    test_feature = np.array(test_dataset[0])

    word_distribution = np.matmul(test_representation, topic_word_distribution)
    corpus_log_prob, corpus_length = 0, 0
    for doc_index in range(len(test_feature)):
        doc = test_feature[doc_index]
        doc_len = np.sum(doc)
        doc_log_prob = 0
        for index in range(len(doc)):
            token_num = doc[index]
            token_probability = word_distribution[doc_index][index]
            if token_num == 0:
                token_log_prob = 0
            else:
                token_log_prob = math.log(token_probability) * token_num
            doc_log_prob += token_log_prob
        corpus_log_prob += doc_log_prob
        corpus_length += doc_len
    perplexity = math.exp(-1*corpus_log_prob/corpus_length)
    write_info = []
    for key in cv_para:
        write_info.append([key, cv_para[key]])
    write_info.append([])
    write_info.append(['perplexity', perplexity])
    with open(perplexity_csv.format(dataset_name, timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(write_info)


def write_representation(model, train_dataset, test_dataset, cv_para):
    dataset_name = cv_para['dataset_name']
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    train_feature = torch.FloatTensor(np.array(train_dataset[0])).to(device)
    train_representation = model.ntm.get_topic_distribution(train_feature, 'inference').detach().to(
        'cpu').numpy()
    test_feature = torch.FloatTensor(np.array(test_dataset[0])).to(device)
    test_representation = model.ntm.get_topic_distribution(test_feature, 'inference').detach().to(
        'cpu').numpy()
    pickle.dump([train_representation, test_representation],
                open(representation_pkl.format(dataset_name, timestamp), 'wb'))
