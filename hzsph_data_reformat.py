import random
import numpy as np
import os
import torch
import csv
import pkuseg
from config import logger, hzsph_cache, cn_CLS_token, diagnosis_map, device, topic_model_first_emr_parse_list, \
    topic_model_admission_parse_list, cache_dir, args, neural_network_first_emr_parse_list, \
    neural_network_admission_parse_list, semi_structure_admission_path, \
    emr_parse_file_path, reorganize_first_emr_path, hzsph_data_file_template, integrate_file_name, parse_list, \
    skip_word_set, cn_PAD_token
import pickle
from itertools import islice
from transformers import BertModel, BertTokenizer


def hzsph_load_data(read_from_cache, vocab_size_ntm, cut_length):
    if read_from_cache and os.path.exists(hzsph_cache):
        five_fold_data, word_index_map = pickle.load(open(hzsph_cache, 'rb'))
    else:
        word_index_map, reformat_data = hzsph_bag_of_word_reorganize(vocab_size_ntm, cut_length)
        embedding = hzsph_data_embedding()
        index_list = [i for i in range(len(reformat_data))]
        random.shuffle(index_list)
        five_fold_feature = hzsph_five_fold_datasets(reformat_data, index_list)
        five_fold_embedding = hzsph_five_fold_datasets(embedding, index_list)
        five_fold_data = list()
        for fold_feature, fold_embedding in zip(five_fold_feature, five_fold_embedding):
            fold_info = list()
            for item_feature, item_embedding in zip(fold_feature, fold_embedding):
                assert item_feature[0] == item_embedding[0]
                key, feature, diagnosis, embedding, info_str = \
                    item_feature[0], item_feature[1], item_feature[2], item_embedding[1], item_embedding[3]
                fold_info.append([key, feature, embedding, diagnosis, info_str])
            five_fold_data.append(fold_info)
        pickle.dump((five_fold_data, word_index_map), open(hzsph_cache, 'wb'))
    logger.info('data loaded')
    return five_fold_data, word_index_map


def hzsph_five_fold_datasets(data, shuffle_index_list):
    """
    五折交叉验证
    """
    data_list = []
    for key in data:
        data_list.append([key] + list(data[key]))

    shuffled_data_list = []
    for index in shuffle_index_list:
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


def hzsph_word_index_convert(data, vocab_size, cut_length):
    data_list = []
    for key in data:
        data_list.append([key, data[key][0], diagnosis_map[data[key][1]]])
    content_list = [item[1] for item in data_list]
    model = pkuseg.pkuseg(model_name='medicine')
    # HanLP = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_SMALL)
    # tokenize_data = HanLP(content_list)
    tokenize_data = list()
    for item in content_list:
        tokenize_data.append(model.cut(item))

    if cut_length > 0:
        for i in range(len(tokenize_data)):
            tokenize_data[i] = tokenize_data[i][:cut_length]

    reformat_data = list()
    for i in range(len(tokenize_data)):
        reformat_data.append([data_list[i][0], tokenize_data[i], data_list[i][2]])

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


def hzsph_bag_of_word_reorganize(vocab_size, cut_length):
    admission_data_dict, first_emr_record_dict = hzsph_preliminary_load_data(read_from_cache=False)
    data = hzsph_reconstruct_data(admission_data_dict, first_emr_record_dict, topic_model_admission_parse_list,
                                  topic_model_first_emr_parse_list)
    word_index_map, reformat_data = hzsph_word_index_convert(data, vocab_size, cut_length)
    return word_index_map, reformat_data


def hzsph_data_embedding():
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base', cache_dir=cache_dir)
    model = BertModel.from_pretrained('hfl/chinese-macbert-base', cache_dir=cache_dir).to(device)
    padding_token = tokenizer.convert_tokens_to_ids([cn_PAD_token])
    assert len(padding_token) == 1
    admission_data_dict, first_emr_record_dict = hzsph_preliminary_load_data()
    data = hzsph_reconstruct_data(admission_data_dict, first_emr_record_dict, neural_network_admission_parse_list,
                                  neural_network_first_emr_parse_list)
    embedding_data = dict()
    length_list = []
    for key in data:
        info_str, diagnosis = data[key]
        token = tokenizer(cn_CLS_token + ' ' + info_str)['input_ids']
        # token_str = tokenizer.tokenize(cn_CLS_token + ' ' + info_str)
        length = len(token)
        length_list.append(length)
        if length > 512:
            token = token[:512]
            print('{} input id len is larger than 512'.format(key))
        embedding_data[key] = model(torch.LongTensor([token]).to(device))['last_hidden_state'][0][0].\
            detach().cpu().numpy(), diagnosis_map[diagnosis], info_str

    print('average len {}'.format(np.average(length_list)))
    print('max len {}'.format(np.max(length_list)))
    print('min len {}'.format(np.min(length_list)))
    return embedding_data


def hzsph_reconstruct_data(admission_data_dict, first_emr_record_dict, admission_parse_list, first_emr_parse_list):
    data = dict()
    for key in admission_data_dict:
        if key not in first_emr_record_dict:
            # print('{} not in first emr record'.format(key))
            continue
        admission_str = hzsph_reconstruct_str(admission_parse_list, admission_data_dict[key])
        first_emr_str = hzsph_reconstruct_str(first_emr_parse_list, first_emr_record_dict[key][1])
        data[key] = first_emr_str + ' ; ' + admission_str, key.strip().split('_')[1]
    return data


def hzsph_preliminary_load_data(read_from_cache=True):
    data = hzsph_reorganize_data(integrate_file_name, hzsph_data_file_template, read_from_cache=read_from_cache)
    first_emr_record = hzsph_first_emr_record_reorganize(emr_parse_file_path, data, reorganize_first_emr_path)
    admission_data_dict = hzsph_admission_record_structurize(data)
    hzsph_save_admission_data_dict(admission_data_dict, semi_structure_admission_path)
    return admission_data_dict, first_emr_record


def hzsph_reconstruct_str(parse_order_list, info_dict):
    return_str = ''
    for item in parse_order_list:
        if item in info_dict:
            return_str = return_str + ' ' + item + ': ' + info_dict[item] + ';'
    return return_str


def hzsph_read_emr_parse_file(file_path):
    structure_1, structure_2 = list(), list()
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            if len(line[0]) > 0:
                if ' ' in line[0]:
                    structure_1.append(line[0].strip().split(' '))
                else:
                    structure_1.append([line[0]])
            if len(line) > 1 and len(line[1]) > 0:
                if ' ' in line[0]:
                    structure_2.append(line[1].strip().split(' '))
                else:
                    structure_2.append([line[1]])
    return structure_1, structure_2


def hzsph_first_emr_record_reorganize(file_path, full_data, reorganize_file_path):
    """
    EMR数据中，病程记录相对而言是结构化的非常好的，就目前而言看上去能够比较好的遵循
    """

    structure_1, structure_2 = hzsph_read_emr_parse_file(file_path)
    data = list()
    data_dict = dict()
    for line in full_data:
        if line[1] == '首次病程记录':
            data.append(line)

    for line in data:
        patient_id, doc_type, diagnosis, emr = line
        structured_data, template_type, complete = \
            hzsph_first_emr_structurize(patient_id, emr, structure_1, structure_2)
        if complete:
            assert patient_id+'_'+diagnosis not in data_dict
            data_dict[patient_id+'_'+diagnosis] = doc_type, structured_data, template_type
        # else:
        #     print('incomplete error')

    data_to_write = [['id', 'data_type', 'info_type', 'info content', 'template type']]
    for key in data_dict:
        doc_type, structured_data, template_type = data_dict[key]
        for info_type in structured_data:
            data_to_write.append([key, doc_type, info_type, structured_data[info_type], template_type])
    with open(reorganize_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)
    return data_dict


def hzsph_first_emr_structurize(_, emr, structure_1, structure_2):
    # 首轮清洗，将其中的中医部分删除
    complete_flag = True
    emr_list = emr.split('\n')
    new_emr = ''
    for item in emr_list:
        if '中医' not in item:
            new_emr += (item + '\n')
    new_emr = new_emr.strip()
    # origin_emr = new_emr

    if new_emr.find('病史特点') != -1:
        structure = structure_1
        template_type = 1
    elif new_emr.find('本病特点') != -1:
        structure = structure_2
        template_type = 2
    else:
        raise ValueError('')

    structured_data = dict()
    structured_data['初始信息'] = new_emr[: new_emr.find(structure[0][0])]
    new_emr = new_emr[new_emr.find(new_emr[0]):]

    start_item_index = 0
    while start_item_index < len(structure):
        start_item = structure[start_item_index]
        start_item_alias_name_index = 0
        # 此处不仅不能是符合要求而已，还要比哪个更靠前。
        start_item_alias_index_list = []
        while start_item_alias_name_index < len(start_item):
            alias_name_idx_in_emr = new_emr.find(start_item[start_item_alias_name_index])
            if alias_name_idx_in_emr != -1:
                start_item_alias_index_list.append([alias_name_idx_in_emr, start_item_alias_name_index])
            start_item_alias_name_index += 1

        if len(start_item_alias_index_list) == 0:
            emr_start_index = -1
            emr_start_item_alias_index = -1
        else:
            start_item_alias_index_list.sort(key=lambda x: x[0])
            emr_start_index = start_item_alias_index_list[0][0]
            emr_start_item_alias_index = start_item_alias_index_list[0][1]
        if emr_start_index == -1:
            start_item_index += 1
            continue

        end_item_index = start_item_index + 1
        if end_item_index <= len(structure) - 1:
            emr_end_index = -1
            while end_item_index < len(structure):
                end_item_alias_index_list = []
                end_item = structure[end_item_index]
                end_item_alias_name_index = 0
                while end_item_alias_name_index < len(end_item):
                    alias_name_idx_in_emr = new_emr.find(end_item[end_item_alias_name_index])
                    if alias_name_idx_in_emr != -1:
                        end_item_alias_index_list.append([alias_name_idx_in_emr, end_item_alias_name_index])
                    end_item_alias_name_index += 1
                if len(end_item_alias_index_list) == 0:
                    emr_end_index = -1
                else:
                    end_item_alias_index_list.sort(key=lambda x: x[0])
                    emr_end_index = end_item_alias_index_list[0][0]
                if emr_end_index == -1:
                    if end_item[0] != '诊断依据':
                        # print('{}, missing component: {}'.format(patient_id, end_item[0]))
                        complete_flag = False
                    end_item_index += 1
                else:
                    break
        else:
            emr_end_index = len(new_emr)

        start_item_index = end_item_index

        if emr_end_index != -1:
            structured_data[start_item[0]] = \
                new_emr[emr_start_index + len(start_item[emr_start_item_alias_index]): emr_end_index]
            structured_data[start_item[0]] = \
                structured_data[start_item[0]]
            new_emr = new_emr[emr_end_index:]
        # else:
        #     print('error')

    for key in structured_data:
        structured_data[key] = structured_data[key].replace('\n', '').replace('*', '').replace('\u002a', '')
    return structured_data, complete_flag, template_type


def hzsph_save_admission_data_dict(admission_data_dict, admission_path):
    data_to_write = [['patient_id', '项目', '内容']]
    for patient_id in admission_data_dict:
        for key in admission_data_dict[patient_id]:
            content = admission_data_dict[patient_id][key]
            data_to_write.append([patient_id, key, content])

    with open(admission_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def hzsph_admission_record_structurize(data):
    admission_data_list = []
    admission_data_dict = dict()

    for line in data:
        if line[1] == '入院记录':
            admission_data_list.append(line)
    for line in admission_data_list:
        personalized_parse_list = hzsph_identify_parse_sequence(line[0], line[3], parse_list)
        content_dict = hzsph_content_format(line[0], line[3], personalized_parse_list)
        content_dict['诊断'] = line[2]
        # if line[0] in admission_data_dict:
        #     print('Duplicate')

        for key in content_dict:
            content_dict[key] = content_dict[key].replace('\n', '').replace('*', '').replace('\u002a', '')
        key = line[0]+'_'+line[2]
        admission_data_dict[key] = content_dict
    return admission_data_dict


def hzsph_identify_parse_sequence(_, data_str, sequential_split):
    _ = data_str
    personalized_parse_list, match_dict, parse_map = [], dict(), dict()
    for item in sequential_split:
        parse_map[item[0]] = item

    for i in range(len(sequential_split)):
        match_dict[sequential_split[i][0]] = []

    for i in range(len(sequential_split)):
        for j in range(1, len(sequential_split[i])):
            split_start_name_pattern = sequential_split[i][j]
            match_list = split_start_name_pattern.finditer(data_str)
            if match_list is not None:
                for match in match_list:
                    str_start_start_idx = match.span()[0]
                    match_dict[sequential_split[i][0]].append([str_start_start_idx, split_start_name_pattern])

    parse_order_list = list()
    for key in match_dict:
        for item in match_dict[key]:
            parse_order_list.append([key, item[0], item[1]])
    parse_order_list = sorted(parse_order_list, key=lambda x: x[1])
    new_parse_order_list = []
    for item in parse_order_list:
        if item[1] != -1:
            new_parse_order_list.append(item)
    for item in new_parse_order_list:
        personalized_parse_list.append([item[0], item[2]])

    # for item in match_dict:
    #     if len(match_dict[item]) == 0:
    #         if item not in {"家系谱图", '谈话记录', '月经史', '联系方式', '家庭地址', '身份证号', '宗教', '寄养'} and not \
    #                 ('出院诊断' in origin_emr):
    #             # if patient_id not in {'381501'}:
    #             #     print('{}, item: {} not found'.format(patient_id, item))
    return personalized_parse_list


def hzsph_read_sequential_split_file(path):
    data_to_read = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        for line in islice(csv_reader, 1, None):
            line_ = []
            for item in line:
                if len(item) > 0:
                    line_.append(item)
            data_to_read.append(line_)
    return data_to_read


def hzsph_content_format(_, data_str, sequential_split):
    # 假设顺序排布是固定的，可能有缺失但是不会有顺序变化

    data_dict = dict()
    for item in sequential_split:
        data_dict[item[0]] = ''

    element_start_idx = 0
    while element_start_idx < len(sequential_split) - 1:
        # reinitialize
        str_start_start_idx, str_start_end_idx, str_end_start_idx, str_end_end_idx, len_start_element, \
            element_name = -1, -1, -1, -1, 0, ''

        # find start element
        for i in range(len(sequential_split[element_start_idx][1:])):
            split_start_name_pattern = sequential_split[element_start_idx][i + 1]
            match = split_start_name_pattern.search(data_str)
            if match is not None:
                str_start_start_idx, str_start_end_idx = match.regs[0]
            if str_start_start_idx != -1:
                element_name = sequential_split[element_start_idx][0]
                break

        if str_start_start_idx == -1:
            # print('{}, {}, not found'.format(patient_id, sequential_split[element_start_idx][0]))
            element_start_idx += 1
            continue

        element_end_idx = element_start_idx + 1
        while element_end_idx < len(sequential_split):
            for i in range(len(sequential_split[element_end_idx][1:])):
                split_end_name_pattern = sequential_split[element_end_idx][i + 1]
                match = split_end_name_pattern.search(data_str)
                if match is not None:
                    str_end_start_idx, str_end_end_idx = match.regs[0]
                if str_end_start_idx != -1:
                    element_name = sequential_split[element_start_idx][0]
                    break
            if str_end_start_idx == -1:
                element_end_idx += 1
                continue
            else:
                break

        if str_end_start_idx == -1:
            str_end_start_idx = len(data_str)

        element_start_idx += 1
        data_dict[element_name] += ' ' + data_str[str_start_end_idx: str_end_start_idx]
        data_str = data_str[str_end_start_idx:]

    return data_dict


def hzsph_reorganize_data(file_path, file_path_template, read_from_cache=False):
    if read_from_cache is True and os.path.exists(file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8-sig', newline='') as f:
            csv_reader = csv.reader(f)
            for line in islice(csv_reader, 1, None):
                data.append(line)
    else:
        folder_name_list = ['双相', '抑郁', '焦虑障碍']
        file_name_list = ['入院记录', '出院记录', '病程记录', '首次查房记录', '首次病程记录']
        data_to_write = [['患者ID', '入院类型', '诊断', '内容']]
        for diagnosis in folder_name_list:
            for file_type in file_name_list:
                path = file_path_template.format(diagnosis, file_type)
                with open(path, 'r', encoding='gb18030') as file:
                    csv_reader = csv.reader(file)
                    for line in islice(csv_reader, 1, None):
                        patient_id = line[0].strip()
                        data_type = line[1].strip()
                        content = line[2].strip()
                        assert data_type == file_type
                        data_to_write.append([patient_id, data_type, diagnosis, content])

        with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerows(data_to_write)
        data = data_to_write[1:]
    return data


def hzsph_re_save_data(file_path_template, save_, re_save=False):
    folder_name_list = ['双相', '抑郁', '焦虑障碍']
    file_name_list = ['入院记录', '出院记录', '病程记录', '首次查房记录', '首次病程记录']
    for diagnosis in folder_name_list:
        for file_type in file_name_list:
            write_path = os.path.join(save_, diagnosis + '_' + file_type + '.csv')
            if re_save is False and os.path.exists(write_path):
                return True
            path = file_path_template.format(diagnosis, file_type)
            data_to_rewrite = []
            with open(path, 'r', encoding='gb18030') as file:
                csv_reader = csv.reader(file)
                for line in csv_reader:
                    data_to_rewrite.append(line)

            with open(write_path, 'w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerows(data_to_rewrite)
    return True


def main():
    vocab_size_ntm = args['vocab_size_ntm']
    read_from_cache = args['read_from_cache']
    cut_length = args['cut_length']
    hzsph_load_data(read_from_cache, vocab_size_ntm, cut_length)
    print('')


if __name__ == '__main__':
    main()
