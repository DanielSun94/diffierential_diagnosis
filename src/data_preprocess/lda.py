import hanlp
from config import topic_model_first_emr_parse_list, topic_model_admission_parse_list, diagnosis_map, \
    word_count_path, skip_word_set
import pickle
import numpy as np
import os
from util import five_fold_datasets, dataset_format, evaluation
from data_reformat import load_data, reconstruct_data
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation


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


def train(topic_number, vocab_size):
    word_index_map, reformat_data = bag_of_word_reorganize(vocab_size)
    five_fold_data = five_fold_datasets(reformat_data)
    accuracy_list = []
    for i in range(5):
        # print('iter: {}'.format(i))
        test_dataset, train_dataset = five_fold_data[i], []
        for j in range(5):
            if i != j:
                for item in five_fold_data[j]:
                    train_dataset.append(item)
        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        lda = LatentDirichletAllocation(n_components=topic_number, random_state=0)
        lda.fit(train_dataset[0])
        train_representation = lda.transform(train_dataset[0])
        test_representation = lda.transform(test_dataset[0])
        mlp_model = MLPClassifier(hidden_layer_sizes=(), max_iter=2000)
        mlp_model.fit(train_representation, train_dataset[1])
        prediction = mlp_model.predict_proba(test_representation)
        accuracy = evaluation(prediction, test_dataset[1])
        # print('iter {}, accuracy: {}'.format(i, accuracy))
        accuracy_list.append(accuracy)
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    for topic_num in 5, 10, 20, 30:
        for vocab_size in 5000, 10000, 15000, 20000, 30000:
            print('topic number: {}, vocab_number: {}'.format(topic_num, vocab_size))
            train(topic_num, vocab_size)


if __name__ == '__main__':
    main()
