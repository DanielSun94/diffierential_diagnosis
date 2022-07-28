import csv
import os
import pickle
from itertools import islice
# from transformers import DebertaTokenizer, DebertaModel
from transformers import LongformerModel, LongformerTokenizer
import numpy as np
import torch
import random
from config import mimic_iii_note_path, mimic_iii_diagnoses, mimic_iii_cache_0, device, \
    mimic_iii_cache_2, mimic_iii_cache_1, mimic_iii_cache_3, args, cache_dir
from sklearn.feature_extraction.text import TfidfVectorizer


def read_mimic_data(read_from_cache=True):
    # read diagnosis
    # 此处我们只评估第一诊断，与七院数据采取一样的策略
    # 经过查阅文档，确认MIMIC-III的数据集是根据优先级排列的，因此即取seq_num为1的item
    if read_from_cache and os.path.exists(mimic_iii_cache_0):
        diagnosis_dict, report_dict = pickle.load(open(mimic_iii_cache_0, 'rb'))
        return diagnosis_dict, report_dict

    diagnosis_dict = dict()
    report_dict = dict()
    with open(mimic_iii_diagnoses, 'r', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, seq_num, icd_9_code = line[1:]
            if seq_num == '1':
                identifier = patient_id+'-'+visit_id
                assert identifier not in diagnosis_dict
                diagnosis_dict[identifier] = icd_9_code

    with open(mimic_iii_note_path, 'r', newline='') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id, category, description, content = line[1], line[2], line[6], line[7], line[10]
            if category.lower() == 'discharge summary':
                identifier = patient_id + '-' + visit_id
                # assert identifier not in report_dict
                report_dict[identifier] = content

    pickle.dump((diagnosis_dict, report_dict), open(mimic_iii_cache_0, 'wb'))
    return diagnosis_dict, report_dict


def note_extract(report_dict):
    extract_list = [
        ['HISTORY OF PRESENT ILLNESS:', 'history of the present illness:'],
        ['PAST MEDICAL HISTORY:'],
        ['MEDICATIONS ON ADMISSION:', 'admission medications:', 'past medical history:'],
        ['ALLERGIES:'],
        ['SOCIAL HISTORY:'],
        ['FAMILY HISTORY:'],
        ['PHYSICAL EXAM AT TIME OF ADMISSION:', 'PHYSICAL EXAM:', 'physical examination:'],
        ['intensive care unit course:'],
        ['BRIEF SUMMARY OF HOSPITAL COURSE:', 'hospital course:'],
        ['LABORATORY STUDIES:', 'LABORATORY DATA:'],
    ]
    key_info_set = {
        'history of present illness:', 'past medical history:', 'medications on admission:',
        'physical exam at time of admission:'
    }
    complete_data_dict = key_info_detect(report_dict, extract_list, key_info_set)
    emr_dict = text_reorganize(complete_data_dict, report_dict, key_info_set)

    length_list = []
    for key in emr_dict:
        content_length = len(emr_dict[key].split(' '))
        length_list.append(content_length)
    print('data size: {}, average token length: {}'.format(len(length_list), np.average(length_list)))
    return emr_dict


def text_reorganize(complete_data_dict, report_dict, key_info_set):
    emr_dict = dict()
    for identifier in complete_data_dict:
        context_index_dict = complete_data_dict[identifier]
        content = report_dict[identifier]
        new_content = ''

        index_list = []
        for key in context_index_dict:
            if context_index_dict[key][0] != -1:
                index_list.append([context_index_dict[key][0], context_index_dict[key][1], key])
        index_list = sorted(index_list, key=lambda x: x[0])
        for i in range(len(index_list)-1):
            start_index, info_length, info_name = index_list[i]
            info_name = info_name.lower()
            end_index = index_list[i+1][0]
            if info_name in key_info_set:
                new_content += info_name + content[start_index+info_length: end_index] + ' '
        emr_dict[identifier] = new_content.lower()
    return emr_dict


def key_info_detect(report_dict, extract_list, key_info_set):
    success_count = 0
    complete_data_dict = dict()
    for identifier in report_dict:
        success_flag = True
        content = report_dict[identifier].lower()
        content_valid_dict = dict()
        content_info_index_dict = dict()
        for item in extract_list:
            content_valid_dict[item[0]] = True
            content_info_index_dict[item[0]] = -1, -1
        for item in extract_list:
            item_type = item[0].lower()
            index = -1
            for specific_item in item:
                specific_item = specific_item.lower()
                index = content.find(specific_item)
                if index != -1:
                    content_info_index_dict[item[0]] = index, len(specific_item)
                    break
            if item_type in key_info_set:
                if index == -1:
                    success_flag = False
                    content_valid_dict[item[0]] = False
        if success_flag:
            success_count += 1
            complete_data_dict[identifier] = content_info_index_dict
        else:
            success_count += 0
    return complete_data_dict


def emr_tokenize(emr_dict):
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=cache_dir)
    # tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base", cache_dir=cache_dir)

    content_info_list = [[key, emr_dict[key]] for key in emr_dict]
    content_list = [item[1] for item in content_info_list]
    index_list = [item[0] for item in content_info_list]
    tokenize_list, current_index = list(), 0
    while current_index < len(content_list):
        if current_index + 128 <= len(content_list):
            end_index = current_index + 128
        else:
            end_index = len(content_list)
        batch = content_list[current_index: end_index]

        batch_token = tokenizer(batch).data['input_ids']
        for item in batch_token:
            tokenize_list.append(item)
        current_index = end_index

    assert len(tokenize_list) == len(content_list)
    tokenize_dict = dict()
    for index, token in zip(index_list, tokenize_list):
        tokenize_dict[index] = token
    return tokenize_dict


def emr_embedding(tokenize_dict):
    model = LongformerModel.from_pretrained("allenai/longformer-base-4096", cache_dir=cache_dir).to(device)
    # model = DebertaModel.from_pretrained("microsoft/deberta-base", cache_dir=cache_dir).to(device)
    content_info_list = [[key, tokenize_dict[key]] for key in tokenize_dict]
    token_list = [item[1] for item in content_info_list]
    index_list = [item[0] for item in content_info_list]

    representation_list, current_index = list(), 0
    while current_index < len(token_list):
        if current_index + 4 <= len(token_list):
            end_index = current_index + 4
        else:
            end_index = len(token_list)
        batch = token_list[current_index: end_index]
        for i in range(len(batch)):
            if len(batch[i]) > 4096:
                batch[i] = batch[i][:4096]
            else:
                batch[i] = batch[i] + (4096 - len(batch[i])) * [0]

        representations = model(torch.LongTensor(batch).to(device))['last_hidden_state'].detach().cpu().numpy()[:, 0, :]
        for item in representations:
            representation_list.append(item)
        current_index = end_index
        print(current_index)

    assert len(representation_list) == len(token_list)
    representation_dict = dict()
    for index, token in zip(index_list, representation_list):
        representation_dict[index] = token
    return representation_dict


def load_mimic_raw_data(read_from_cache):
    if read_from_cache and os.path.exists(mimic_iii_cache_1):
        emr_dict, tokenize_dict, diagnosis_dict, report_dict = pickle.load(open(mimic_iii_cache_1, 'rb'))
    else:
        diagnosis_dict, report_dict = read_mimic_data()
        emr_dict = note_extract(report_dict)
        tokenize_dict = emr_tokenize(emr_dict)
        pickle.dump((emr_dict, tokenize_dict, diagnosis_dict, report_dict), open(mimic_iii_cache_1, 'wb'))

    print('tokenize finished')
    if read_from_cache and os.path.exists(mimic_iii_cache_2):
        embedding_dict = pickle.load(open(mimic_iii_cache_2, 'rb'))
    else:
        embedding_dict = emr_embedding(tokenize_dict)
        pickle.dump(embedding_dict, open(mimic_iii_cache_2, 'wb'))
    return emr_dict, tokenize_dict, diagnosis_dict, report_dict, embedding_dict


def mimic_data_reorganize(diagnosis_dict, tokenize_dict, embedding_dict, top_n_disease, max_token,
                          read_from_cache, cut_length):
    if read_from_cache and os.path.exists(mimic_iii_cache_3):
        target_disease_map, bag_of_word_dict, token_idx_map, patient_info_dict, shuffled_data = \
            pickle.load(open(mimic_iii_cache_3, 'rb'))
    else:
        target_disease_map = top_disease_select(diagnosis_dict, top_n_disease)
        bag_of_word_dict, token_idx_map = bag_of_words_generation(tokenize_dict, max_token, cut_length)
        patient_info_dict = dict()
        for identifier in diagnosis_dict:
            diagnosis = diagnosis_dict[identifier]
            if diagnosis not in target_disease_map:
                continue
            if (identifier not in embedding_dict) or (identifier not in bag_of_word_dict):
                continue
            embedding = embedding_dict[identifier]
            bag_of_word = bag_of_word_dict[identifier]
            patient_info_dict[identifier] = [target_disease_map[diagnosis], embedding, bag_of_word]

        shuffled_data = mimic_five_fold_generation(patient_info_dict)
        pickle.dump((target_disease_map, bag_of_word_dict, token_idx_map, patient_info_dict, shuffled_data),
                    open(mimic_iii_cache_3, 'wb'))
    return shuffled_data, target_disease_map, bag_of_word_dict, token_idx_map, patient_info_dict


def mimic_five_fold_generation(patient_info_dict):
    index_list = [i for i in range(len(patient_info_dict))]
    patient_info_list = [(patient_info_dict[identifier], identifier) for identifier in patient_info_dict]
    random.shuffle(index_list)
    shuffled_list = []
    for index in index_list:
        disease_index, embedding, bag_of_word = patient_info_list[index][0]
        identifier = patient_info_list[index][0]
        shuffled_list.append([identifier, bag_of_word, embedding, disease_index])

    fold_size = len(shuffled_list) // 5
    shuffled_data = [
        shuffled_list[0: fold_size],
        shuffled_list[fold_size: fold_size * 2],
        shuffled_list[fold_size * 2: fold_size * 3],
        shuffled_list[fold_size * 3: fold_size * 4],
        shuffled_list[fold_size * 4:],
    ]
    return shuffled_data


def bag_of_words_generation(tokenize_dict, vocab_size, cut_length):
    vectorizer = TfidfVectorizer(max_features=vocab_size)
    token_id_list = []
    for key in tokenize_dict:
        token_id_str_list = [str(item) for item in tokenize_dict[key]]
        if cut_length > 0:
            token_id_str_list = token_id_str_list[:cut_length]

        token_id_str = ' '.join(token_id_str_list)
        token_id_list.append(token_id_str)
    vectorizer.fit(token_id_list)
    target_token_list = vectorizer.get_feature_names_out()
    target_token_set = set([int(item) for item in target_token_list])
    token_idx_map = dict()
    for token in target_token_set:
        token_idx_map[token] = len(token_idx_map)

    bag_of_word_dict = dict()
    for key in tokenize_dict:
        token_list = tokenize_dict[key]
        bag_of_word_representation = np.zeros(vocab_size)
        for token in token_list:
            if token in token_idx_map:
                bag_of_word_representation[token_idx_map[token]] += 1
        bag_of_word_dict[key] = bag_of_word_representation
    return bag_of_word_dict, token_idx_map


def top_disease_select(diagnosis_dict, top_k_disease):
    disease_count_dict = {}
    for key in diagnosis_dict:
        icd_code = diagnosis_dict[key]
        if icd_code not in disease_count_dict:
            disease_count_dict[icd_code] = 0
        disease_count_dict[icd_code] += 1

    disease_count_list = [[key, disease_count_dict[key]] for key in disease_count_dict]
    disease_count_list = sorted(disease_count_list, key=lambda x: x[1], reverse=True)
    target_disease_dict = dict()
    for i in range(top_k_disease):
        target_disease_dict[disease_count_list[i][0]] = len(target_disease_dict)
    return target_disease_dict


def mimic_load_data(vocab_size, diagnosis_size, read_from_cache, cut_length):
    emr_dict, tokenize_dict, diagnosis_dict, report_dict, embedding_dict = load_mimic_raw_data(read_from_cache)
    shuffled_data, target_disease_map, bag_of_word_dict, token_idx_map, patient_info_dict = \
        mimic_data_reorganize(diagnosis_dict, tokenize_dict, embedding_dict, diagnosis_size, vocab_size,
                              read_from_cache, cut_length)
    return shuffled_data, token_idx_map


def main():
    diagnosis_size = args['diagnosis_size']
    vocab_size_ntm = args['vocab_size_ntm']
    read_from_cache = args['read_from_cache']
    cut_length = args['cut_length']
    mimic_load_data(vocab_size_ntm, diagnosis_size, read_from_cache, cut_length)
    print('')


if __name__ == '__main__':
    main()
