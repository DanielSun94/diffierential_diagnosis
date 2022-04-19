import os
import csv
from itertools import islice


def main():
    integrate_file_name = os.path.abspath('../../data/data_utf_8/integrate_data.csv')
    admission_parse_file_path = os.path.abspath('../../data/data_utf_8/入院记录解析序列.csv')
    emr_parse_file_path = os.path.abspath('../../data/data_utf_8/病程记录解析序列.csv')

    semi_structure_admission_path = os.path.abspath('../../data/data_utf_8/半结构化入院记录.csv')
    data_file_template = os.path.abspath('../../data/origin_data/{}/{}.csv')
    save_folder = os.path.abspath('../../data/data_utf_8')
    re_save_data(data_file_template, save_folder)

    data = reorganize_data(integrate_file_name, data_file_template)
    first_emr_record(emr_parse_file_path, data)
    # admission_data_dict = admission_record_structurize(data, admission_parse_file_path)
    # print_admission_data_dict(admission_data_dict, semi_structure_admission_path)


def read_emr_parse_file(file_path):
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


def first_emr_record(emr_parse_file_path, full_data):
    """
    EMR数据中，病程记录相对而言是结构化的非常好的，就目前而言看上去能够比较好的遵循
    """

    structure_1, structure_2 = read_emr_parse_file(emr_parse_file_path)
    data = list()
    for line in full_data:
        if line[1] == '首次病程记录':
            data.append(line)

    for line in data:
        patient_id, doc_type, diagnosis, emr = line
        _emr_structurize(patient_id, emr, structure_1, structure_2)
    return data


def _emr_structurize(patient_id, emr, structure_1, structure_2):
    origin_emr = emr
    if emr.find('病史特点') != -1:
        structure = structure_1
    elif emr.find('本病特点') != -1:
        structure = structure_2
    else:
        raise ValueError('')

    structured_data = dict()
    structured_data['初始信息'] = emr[: emr.find(structure[0][0])]
    emr = emr[emr.find(emr[0]):]

    index = 0
    while index < len(structure):
        item = structure[index]
        start_index, end_index = -1, -1
        item_start_index = 0
        # 此处不仅不能是符合要求而已，还要比哪个更靠前。
        start_index_list = []
        while item_start_index < len(item):
            start_index = emr.find(item[item_start_index])
            if start_index != -1:
                break
            else:
                item_start_index += 1
        if start_index == -1:
            index += 1
            continue

        index_2 = index + 1
        if index < len(structure) - 1:
            while index_2 < len(structure):
                end_item = structure[index_2]
                item_end_index = 0
                while item_end_index < len(end_item):
                    end_index = emr.find(end_item[item_end_index])
                    if end_index != -1:
                        break
                    else:
                        item_end_index += 1
                if end_index == -1:
                    if end_item[0] != '诊断依据':
                        print('{}, missing component: {}'.format(patient_id, end_item[0]))
                    index_2 += 1
                else:
                    break
        else:
            end_index = len(emr)

        index = index_2

        if end_index != -1:
            structured_data[item[0]] = emr[start_index + len(item[item_start_index]): end_index]
            emr = emr[end_index:]
        else:
            print('error')
    return structured_data


def print_admission_data_dict(admission_data_dict, semi_structure_admission_path):
    data_to_write = [['patient_id', '项目', '内容']]
    for patient_id in admission_data_dict:
        for key in admission_data_dict[patient_id]:
            content = admission_data_dict[patient_id][key]
            data_to_write.append([patient_id, key, content])

    with open(semi_structure_admission_path, 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(data_to_write)


def admission_record_structurize(data, sequential_split_file_path):
    admission_data_list = []
    admission_data_dict = dict()
    sequential_split = read_sequential_split_file(sequential_split_file_path)

    for line in data:
        if line[1] == '入院记录':
            admission_data_list.append(line)
    for line in admission_data_list:
        content_dict = content_format(line[0], line[3], sequential_split)
        content_dict['诊断'] = line[2]
        if line[0] in admission_data_dict:
            print('Duplicate')
        admission_data_dict[line[0]] = content_dict
    return admission_data_dict


def read_sequential_split_file(path):
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


def content_format(patient_id, data_str, sequential_split):
    # 假设顺序排布是固定的，可能有缺失但是不会有顺序变化

    data_dict = dict()
    for item in sequential_split:
        data_dict[item[0]] = 'None'

    element_start_idx = 0
    while element_start_idx < len(sequential_split) - 1:
        # reinitialize
        str_start_idx, str_end_idx, len_start_element, element_name = -1, -1, 0, ''

        # find start element
        for i in range(len(sequential_split[element_start_idx][1:])):
            split_start_name = sequential_split[element_start_idx][i + 1]
            str_start_idx = data_str.find(split_start_name)
            if str_start_idx != -1:
                element_name = sequential_split[element_start_idx][0]
                len_start_element = len(split_start_name)
                break

        if str_start_idx == -1:
            ignore_info = {'谈话记录', '病史陈述者', '宗教', '入院时间', '联系方式', '家庭地址', '记录时间', '身份证号',
                           '寄养', '家系谱图', '父母近亲婚姻'}
            if sequential_split[element_start_idx][0] not in ignore_info:
                print('{}, {}, not found'.format(patient_id, sequential_split[element_start_idx][0]))
            element_start_idx += 1
            continue
        else:
            str_start_idx = str_start_idx + len_start_element

        element_end_idx = element_start_idx + 1
        while element_end_idx < len(sequential_split):
            for i in range(len(sequential_split[element_end_idx][1:])):
                split_end_name = sequential_split[element_end_idx][i + 1]
                str_end_idx = data_str.find(split_end_name)
                if str_end_idx != -1:
                    break
            if str_end_idx == -1:
                element_end_idx += 1
                continue
            else:
                break

        if str_end_idx == -1:
            str_end_idx = len(data_str)

        element_start_idx += 1
        data_dict[element_name] = data_str[str_start_idx: str_end_idx]
        data_str = data_str[str_end_idx:]

    return data_dict


def reorganize_data(file_path, file_path_template, read_from_cache=True):
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


def re_save_data(file_path_template, save_folder, re_save=False):
    folder_name_list = ['双相', '抑郁', '焦虑障碍']
    file_name_list = ['入院记录', '出院记录', '病程记录', '首次查房记录', '首次病程记录']
    for diagnosis in folder_name_list:
        for file_type in file_name_list:
            write_path = os.path.join(save_folder, diagnosis + '_' + file_type + '.csv')
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


if __name__ == '__main__':
    main()
