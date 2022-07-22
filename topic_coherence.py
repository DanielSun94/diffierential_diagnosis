from math import log

import torch

from train import one_cv_train
from config import args, logger, model_name_template, lda_save_template
from util import dataset_selection, dataset_format
import os
import numpy as np
from model import TopicAwareNTM
import pickle
from sklearn.decomposition import LatentDirichletAllocation

phase = 'ntm_inference'


def main():
    batch_size, hidden_size_ntm, model, tau, diagnosis_size, topic_number_ntm, learning_rate, vocab_size_ntm, \
        epoch_number, device, dataset_name, repeat_time, read_from_cache, test_set_num, experiment_type, vocab_size = \
        args['batch_size'], args['hidden_size_ntm'], args['model'], args['tau'], args['diagnosis_size'], \
        args['topic_number_ntm'], args['learning_rate'], args['vocab_size_ntm'],  args['epoch_number'], \
        args['device'], args['dataset_name'], args['repeat_time'], args['read_from_cache'],\
        args['test_set_num'], args['experiment_type'], args['vocab_size_lda']

    for key in args:
        logger.info('{}: {}'.format(key, args[key]))

    if phase == 'ntm_train':
        ntm_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number, device,
                     tau, model, diagnosis_size, read_from_cache, test_set_num)
    if phase == 'lda_analysis':
        for dataset_name in 'hzsph', 'mimic-iii':
            lda, representation, word_index_map = \
                lda_analysis(dataset_name, vocab_size, diagnosis_size, read_from_cache, topic_number=10)
            npmi = lda_topic_coherence_analysis(lda, dataset_name, vocab_size, diagnosis_size, read_from_cache)
            logger.info('lda, dataset: {}, npmi: {}'.format(dataset_name, npmi))

    if phase == 'ntm_inference':
        for dataset_name in 'hzsph', 'mimic-iii':
            model_config_list = [[0, 0, 0], [0, 0, 0.1], [0, 0.1, 0], [0.1, 0, 0], [0.1, 0.1, 0.1]]
            for item in model_config_list:
                topic_coefficient, contrastive_coefficient, similarity_coefficient = item[0], item[1], item[2]
                # topic_coefficient, contrastive_coefficient, similarity_coefficient = 0, 0, 0
                topic_number, model_name = 10, 'rntm'
                model_save_path = model_name_template.format(dataset_name, contrastive_coefficient,
                                                             similarity_coefficient, topic_coefficient)
                if not os.path.exists(model_save_path):
                    continue
                ntm_inference(dataset_name, model_save_path, topic_coefficient, contrastive_coefficient,
                              similarity_coefficient, hidden_size_ntm, topic_number, vocab_size, device, tau,
                              model_name, read_from_cache)


def ntm_inference(dataset_name, model_save_path, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                  hidden_size, topic_number, vocab_size, device, tau, model_name, read_from_cache):

    model = TopicAwareNTM(hidden_size, topic_number, vocab_size, device, tau, model_name)
    model.load_state_dict(torch.load(model_save_path))
    topic_distribution = model.ntm.get_topics().detach().to('cpu').numpy()

    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, 10, read_from_cache)
    dataset = []
    for i in range(5):
        for item in five_fold_data[i]:
            dataset.append(item)
    token_list = dataset_format(dataset)[0]

    if dataset_name == 'hzsph':
        skip_token_set = {9999, 7, 16, 76}
    elif dataset_name == 'mimic-iii':
        skip_token_set = {7503, 28, 1203, 7508, 4683, 653}
    else:
        raise ValueError('')
    # print(word_index_map)

    top_token_lists = select_top_topic_token(topic_distribution, skip_token_set, top_n=10)
    npmi = topic_coherence(top_token_lists, token_list)
    logger.info('ntm cl: {}, kl: {}, dl: {}, dataset: {}, npmi: {}'.format(
        contrastive_coefficient, similarity_coefficient, topic_coefficient, dataset_name, npmi))


def ntm_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number, device,
              tau, model, diagnosis_size, read_from_cache, test_set_num):
    model_config_list = [[0.1, 0, 0]]

    for item in model_config_list:
        topic_coefficient, contrastive_coefficient, similarity_coefficient = item[0], item[1], item[2]
        ntm_coefficient = 1 - topic_coefficient - contrastive_coefficient - similarity_coefficient
        for dataset_name in 'hzsph', 'mimic-iii':
            model_save_path = model_name_template.format(dataset_name, contrastive_coefficient, similarity_coefficient,
                                                         topic_coefficient)
            logger.info('output representation: {}'.format(dataset_name))
            one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                         topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient,
                         device, tau, model, dataset_name, diagnosis_size, read_from_cache, False, False, False,
                         False, test_set_num, True, model_save_path)


def lda_analysis(dataset_name, vocab_size, diagnosis_size, read_from_cache, topic_number):
    if read_from_cache and os.path.exists(lda_save_template.format(dataset_name)):
        lda, representation, word_index_map = pickle.load(open(lda_save_template.format(dataset_name), 'rb'))
    else:
        five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size,
                                                           read_from_cache)
        dataset = []
        for i in range(5):
            for item in five_fold_data[i]:
                dataset.append(item)
        dataset = dataset_format(dataset)
        lda = LatentDirichletAllocation(n_components=topic_number)
        logger.info('start')
        lda.fit(dataset[0])
        logger.info('end')
        representation = lda.transform(dataset[0])
        pickle.dump((lda, representation, word_index_map), open(lda_save_template.format(dataset_name), 'wb'))
    return lda, representation, word_index_map


def lda_topic_coherence_analysis(lda, dataset_name, vocab_size, diagnosis_size,
                                 read_from_cache):
    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size,
                                                       read_from_cache)
    dataset = []
    for i in range(5):
        for item in five_fold_data[i]:
            dataset.append(item)
    token_list = dataset_format(dataset)[0]

    topic_word_distribution = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]

    if dataset == 'hzsph':
        skip_token_set = {5, 6, 10, 22, 24, 30, 33, 12, 40, 45, 47, 58, 68, 76, 9999}
    elif dataset == 'mimic-iii':
        skip_token_set = {}
    else:
        raise ValueError('')
    print(word_index_map)

    top_token_lists = select_top_topic_token(topic_word_distribution, skip_token_set, top_n=10)
    npmi = topic_coherence(top_token_lists, token_list)
    return npmi


def topic_coherence(top_token_lists, token_lists):
    npmi_list, len_token = [], len(token_lists)
    for top_token_list in top_token_lists:
        pmi = 0
        for i in range(len(top_token_lists)-1):
            for j in range(i+1, len(top_token_lists)):
                p_i, p_j, p_ij = 0, 0, 0,
                token_i, token_j = top_token_list[i], top_token_list[j]
                for token_list in token_lists:
                    if token_list[token_i] > 0:
                        p_i += 1
                    if token_list[token_j] > 0:
                        p_j += 1
                    if token_list[token_j] > 0 and token_list[token_i] > 0:
                        p_ij += 1
                if p_ij == 0:
                    return 'error'
                p_i, p_j, p_ij = p_i / len_token, p_j / len_token, p_ij / len_token
                if p_i == 1 or p_j == 1:
                    print('p_i: {}, token: {}, p_j: {}, token: {}'.format(p_i, token_i, p_j, token_j))
                pmi += -1 * log(p_ij / p_i / p_j) / log(p_ij)
        npmi = pmi / len(top_token_lists) / (len(top_token_lists) - 1)
        npmi_list.append(npmi)
    npmi = np.average(npmi_list)
    return npmi


def select_top_topic_token(distributions, skip_token_set, top_n=10):
    top_token_lists = []
    for distribution in distributions:
        token_prob_list = []
        for i in range(len(distribution)):
            token_prob_list.append([i, distribution[i]])
        token_prob_list = sorted(token_prob_list, key=lambda x: x[1], reverse=True)

        selected_tokens = []
        start_index = 0
        while len(selected_tokens) < top_n:
            token = token_prob_list[start_index][0]
            if token not in skip_token_set:
                selected_tokens.append(token)
            start_index += 1
        top_token_lists.append(selected_tokens)
    return top_token_lists


if __name__ == '__main__':
    main()
