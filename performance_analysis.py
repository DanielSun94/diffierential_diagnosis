import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from itertools import islice
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
cm = plt.cm.get_cmap('RdYlBu')
save_folder = os.path.abspath('./')


def read_result():
    file_name_list = ['experiment_4.log', 'experiment_1.log', 'experiment_2.log', 'experiment_3.log']
    result_dict = dict()
    for key_data in 'hzsph', 'mimic-iii':
        result_dict[key_data] = dict()
        for kl in 0, 0.05, 0.1, 0.15:
            result_dict[key_data][kl] = dict()
            for dl in 0, 0.05, 0.1, 0.15:
                result_dict[key_data][kl][dl] = dict()
                for cl in 0, 0.05, 0.1, 0.15:
                    result_dict[key_data][kl][dl][cl] = dict()
                    for test_set_num in 0, 1, 2, 3, 4:
                        result_dict[key_data][kl][dl][cl][test_set_num] = {
                            'accuracy': -1,
                            'macro_precision': -1,
                            'macro_recall': -1,
                            'macro_f1': -1
                        }

    for file in file_name_list:
        with open(os.path.join(save_folder, file), 'r') as f:
            result_data = f.readlines()

            current_info = {'dataset': "", 'kl': -1, 'dl': -1, 'cl': -1, 'accuracy': -1, 'macro_precision': -1,
                            'macro_recall': -1, 'macro_f1': -1, 'test_set_num': -1}

            first_flag = True
            for line in result_data:
                if 'hyperparameter analysis' in line:
                    if not first_flag:
                        for key in current_info:
                            assert current_info[key] != -1 and current_info[key] != ''
                        assert current_info['cl'] in {0, 0.05, 0.1, 0.15}
                        assert current_info['kl'] in {0, 0.05, 0.1, 0.15}
                        assert current_info['dl'] in {0, 0.05, 0.1, 0.15}
                        result_dict[current_info['dataset']][current_info['kl']][current_info['dl']][
                            current_info['cl']][current_info['test_set_num']] = \
                            {'accuracy': current_info['accuracy'], 'macro_precision': current_info['macro_precision'],
                             'macro_recall': current_info['macro_recall'], 'macro_f1': current_info['macro_f1']}
                    first_flag = False

                    current_info = {'dataset': "", 'kl': -1, 'dl': -1, 'cl': -1, 'accuracy': -1, 'macro_precision': -1,
                                    'macro_recall': -1, 'macro_f1': -1, 'test_set_num': -1}
                    if 'mimic-iii' in line:
                        current_info['dataset'] = 'mimic-iii'
                    if 'hzsph' in line:
                        current_info['dataset'] = 'hzsph'
                if 'similarity_coefficient: ' in line:
                    start_index = line.find('similarity_coefficient: ') + len('similarity_coefficient: ')
                    kl_coefficient = float(line[start_index:].strip())
                    current_info['kl'] = kl_coefficient
                if 'topic_coefficient: ' in line:
                    start_index = line.find('topic_coefficient: ') + len('topic_coefficient: ')
                    dl_coefficient = float(line[start_index:].strip())
                    current_info['dl'] = dl_coefficient
                if 'contrastive_coefficient: ' in line:
                    start_index = line.find('contrastive_coefficient: ') + len('contrastive_coefficient: ')
                    cl_coefficient = float(line[start_index:].strip())
                    current_info['cl'] = cl_coefficient
                if 'accuracy: ' in line:
                    start_index = line.find('accuracy: ') + len('accuracy: ')
                    accuracy = float(line[start_index:].strip())
                    current_info['accuracy'] = accuracy
                if 'macro_recall: ' in line:
                    start_index = line.find('macro_recall: ') + len('macro_recall: ')
                    macro_recall = float(line[start_index:].strip())
                    current_info['macro_recall'] = macro_recall
                if 'macro_precision: ' in line:
                    start_index = line.find('macro_precision: ') + len('macro_precision: ')
                    macro_precision = float(line[start_index:].strip())
                    current_info['macro_precision'] = macro_precision
                if 'macro_f1: ' in line:
                    start_index = line.find('macro_f1: ') + len('macro_f1: ')
                    macro_f1 = float(line[start_index:].strip())
                    current_info['macro_f1'] = macro_f1
                if 'test_set_num: ' in line:
                    start_index = line.find('test_set_num: ') + len('test_set_num: ')
                    test_set_num = int(line[start_index:].strip())
                    assert test_set_num in {0, 1, 2, 3, 4}
                    current_info['test_set_num'] = test_set_num
    return result_dict


def print_result(result_dict):
    detail_data_to_write = [['dataset', 'knowledge distillation loss coefficient', 'topic diversity loss coefficient',
                             'contrastive loss coefficient', 'test_num_set', 'metric', 'value']]
    for key_data in result_dict:
        for kl in result_dict[key_data]:
            for dl in result_dict[key_data][kl]:
                for cl in result_dict[key_data][kl][dl]:
                    for test_set_num in result_dict[key_data][kl][dl][cl]:
                        for metric in result_dict[key_data][kl][dl][cl][test_set_num]:
                            value = result_dict[key_data][kl][dl][cl][test_set_num][metric]
                            detail_data_to_write.append([key_data, kl, dl, cl, test_set_num, metric, value])
    with open(os.path.join(save_folder, 'predictive_result_detail.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(detail_data_to_write)

    merge_data_to_write = [['dataset', 'knowledge distillation loss coefficient', 'topic diversity loss coefficient',
                            'contrastive loss coefficient', 'accuracy', 'precision', 'recall', 'f1', 'num']]
    for key_data in result_dict:
        for kl in result_dict[key_data]:
            for dl in result_dict[key_data][kl]:
                for cl in result_dict[key_data][kl][dl]:
                    acc, precision, recall, f1, acc_num, precision_num, recall_num, f1_num = 0, 0, 0, 0, 0, 0, 0, 0
                    for test_set_num in result_dict[key_data][kl][dl][cl]:
                        for metric in result_dict[key_data][kl][dl][cl][test_set_num]:
                            value = result_dict[key_data][kl][dl][cl][test_set_num][metric]
                            if metric == 'accuracy' and value != -1:
                                acc += value
                                acc_num += 1
                            if metric == 'macro_precision' and value != -1:
                                precision += value
                                precision_num += 1
                            if metric == 'macro_recall' and value != -1:
                                recall += value
                                recall_num += 1
                            if metric == 'macro_f1' and value != -1:
                                f1 += value
                                f1_num += 1

                    merge_data_to_write.append([key_data, kl, dl, cl,
                                                acc / acc_num if acc_num != 0 else -1,
                                                precision / precision_num if precision_num != 0 else -1,
                                                recall / recall_num if recall_num != 0 else -1,
                                                f1 / f1_num if f1_num != 0 else -1, acc_num])
    with open(os.path.join(save_folder, 'predictive_result_merge.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(merge_data_to_write)


def plot_result():
    with open(os.path.join(save_folder, 'predictive_result_merge_plot.csv'), 'r', newline='', encoding='utf-8-sig') as f:
        csv_reader = csv.reader(f)
        hzsph_kl_list, hzsph_cl_list, hzsph_dl_list, hzsph_value_list = [], [], [], []
        mimic_kl_list, mimic_cl_list, mimic_dl_list, mimic_value_list = [], [], [], []
        for line in islice(csv_reader, 1, None):
            kl, dl, cl, value = float(line[1]), float(line[2]), float(line[3]), float(line[4])
            kl = 0.15 if kl == 0.2 else kl
            dl = 0.15 if dl == 0.2 else dl
            cl = 0.15 if cl == 0.2 else cl

            if line[0] == 'hzsph':
                hzsph_kl_list.append(kl)
                hzsph_dl_list.append(dl)
                hzsph_cl_list.append(cl)
                hzsph_value_list.append(value)
            elif line[0] == 'mimic-iii':
                mimic_kl_list.append(kl)
                mimic_dl_list.append(dl)
                mimic_cl_list.append(cl)
                mimic_value_list.append(value)
            else:
                ValueError('')

    fig = plt.figure()
    fig.set_size_inches(7.12, 3.2)
    ax_0 = fig.add_subplot(121, projection='3d')

    # syntax for 3-D projection

    # defining axes
    z = np.array(hzsph_kl_list, dtype=float)
    x = np.array(hzsph_dl_list, dtype=float)
    y = np.array(hzsph_cl_list, dtype=float)
    c = np.array(hzsph_value_list, dtype=float)
    c = c - np.min(c)
    ax_0.scatter(x, y, z, c=c)
    ax_0.set_xlabel('β', rotation=330, fontdict={'fontsize': 8})
    ax_0.set_ylabel('γ', rotation=60, fontdict={'fontsize': 8})
    ax_0.set_zlabel('δ', rotation=0, fontdict={'fontsize': 8})
    ax_0.set_title('(a)', y=-0.25, fontdict={'fontsize': 8})
    ax_0.set_xticks([0, 0.05, 0.1, 0.15])
    ax_0.set_yticks([0, 0.05, 0.1, 0.15])
    ax_0.set_zticks([0, 0.05, 0.1, 0.15])

    # syntax for plotting
    # ax_0.set_title('3d Scatter plot geeks for geeks')

    ax_1 = fig.add_subplot(122, projection='3d')

    # syntax for 3-D projection

    # defining axes
    z = np.array(mimic_kl_list, dtype=float)
    x = np.array(mimic_dl_list, dtype=float)
    y = np.array(mimic_cl_list, dtype=float)
    c = np.array(mimic_value_list, dtype=float)
    c = c - np.min(c)
    im = ax_1.scatter(x, y, z, c=c)
    ax_1.set_title('(b)', y=-0.25, fontdict={'fontsize': 8})
    ax_1.set_xlabel('β', rotation=330, fontdict={'fontsize': 8})
    ax_1.set_ylabel('γ', rotation=60, fontdict={'fontsize': 8})
    ax_1.set_zlabel('δ', rotation=0, fontdict={'fontsize': 8})
    ax_1.set_xticks([0, 0.05, 0.1, 0.15])
    ax_1.set_yticks([0, 0.05, 0.1, 0.15])
    ax_1.set_zticks([0, 0.05, 0.1, 0.15])


    plt.colorbar(im, ax=ax_1, shrink=0.75, anchor=(1.7, 0.5))

    # syntax for plotting
    # ax_1.set_title('3d Scatter plot geeks for geeks')
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.91, left=0)
    plt.show()
    fig.savefig("hyperparameter_influence.svg")


def main():
    result_dict = read_result()
    print_result(result_dict)
    plot_result()


if __name__ == '__main__':
    main()
