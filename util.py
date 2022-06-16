import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, roc_auc_score
from mimic_data_reformat import load_mimic_data
from hzsph_data_reformat import hzsph_load_data


def dataset_format(dataset):
    feature, label, representation = list(), list(), list()
    for item in dataset:
        if len(item) == 3:
            feature.append(item[1])
            label.append(item[2])
        elif len(item) == 4:
            feature.append(item[1])
            representation.append(item[2])
            label.append(item[3])
        else:
            raise ValueError('')
    if len(dataset[0]) == 3:
        return feature, label
    elif len(dataset[0]) == 4:
        return feature, representation, label
    else:
        raise ValueError('')


def dataset_selection(dataset_name, vocab_size, diagnosis_size, read_from_cache):
    if dataset_name == 'mimic-iii':
        five_fold_data, word_index_map = load_mimic_data(vocab_size, diagnosis_size, read_from_cache)
    elif dataset_name == 'hzsph':
        five_fold_data, word_index_map = hzsph_load_data(read_from_cache, vocab_size)
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
    macro_auc = roc_auc_score(label, predict_prob, multi_class='ovr')
    performance = {
        'accuracy': accuracy,
        'micro_recall': micro_recall,
        'macro_recall': macro_recall,
        'micro_precision': micro_precision,
        'macro_precision': macro_precision,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        # 'micro_auc': micro_auc,
        'macro_auc': macro_auc,
    }
    return performance
