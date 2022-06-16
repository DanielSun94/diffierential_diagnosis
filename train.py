from config import logger, args, classify_weight_csv, diagnosis_map, topic_word_distribution_csv
from util import evaluation, dataset_selection, dataset_format
from model import TopicAwareNTM, EHRDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torch import optim
import datetime
import csv
from sklearn.neural_network import MLPClassifier
import numpy as np


def train(batch_size, hidden_size, topic_number, learning_rate, vocab_size, epoch_number, topic_coefficient,
          contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau, model_name, dataset_name,
          diagnosis_size, read_from_cache):
    data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size, read_from_cache)
    performance_list = list()
    logger.info('topic number: {}, vocab size: {}, epoch number: {}'.format(topic_number, vocab_size, epoch_number))
    for i in range(5):
        # print('iter: {}'.format(i))
        test_dataset, train_dataset = data[i], []
        for j in range(5):
            if i != j:
                for item in data[j]:
                    train_dataset.append(item)
        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        training_data = EHRDataset(train_dataset)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        model = TopicAwareNTM(hidden_size, topic_number, vocab_size, device, tau, model_name)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # train_feature = torch.FloatTensor(np.array(train_dataset[0])).to(device)
        # train_representation = model.ntm.get_topic_distribution(train_feature, 'inference').detach().to(
        #     'cpu').numpy()
        for epoch in range(epoch_number):
            epoch_loss = list()
            for batch_data in train_dataloader:
                feature, representation, _, same_class_mat = batch_data
                optimizer.zero_grad()
                output = model(feature, representation, same_class_mat)

                contrastive_loss = output['contrastive_loss'].mean()
                similarity_loss = output['similarity_loss'].mean()
                ntm_loss = output['ntm_loss'].mean()
                topic_word_loss = output['topic_word_loss'].mean()

                loss = ntm_loss * ntm_coefficient + similarity_loss * similarity_coefficient + \
                    contrastive_loss * contrastive_coefficient + topic_word_loss * topic_coefficient
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
                optimizer.step()
                epoch_loss.append(loss.mean().detach().to('cpu').numpy())

            if epoch % 100 == 0:
                performance, _ = calculate_performance(train_dataset, test_dataset, model, device)
                logger.info('epoch: {}, fold: {}, accuracy: {}'.format(epoch, i + 1, performance['accuracy']))
                logger.info('epoch: {}, average loss: {}'.format(epoch, np.average(epoch_loss)))
        performance, classify_model = calculate_performance(train_dataset, test_dataset, model, device)
        # print('iter {}, accuracy: {}'.format(i, accuracy))
        performance_list.append(performance)
        logger.info('fold: {}, accuracy: {}'.format(i+1, performance['accuracy']))
        write_model_info(classify_model, model, word_index_map)
    for key in performance_list[0]:
        score_list = [item[key] for item in performance_list]
        logger.info('average {}: {}'.format(key, np.average(score_list)))


def calculate_performance(train_dataset, test_dataset, model, device):
    if args['classify_model'] == 'nn':
        classify_model = MLPClassifier(hidden_layer_sizes=(), max_iter=6000)
    else:
        raise ValueError('')
    train_feature = torch.FloatTensor(np.array(train_dataset[0])).to(device)
    test_feature = torch.FloatTensor(np.array(test_dataset[0])).to(device)
    train_representation = model.ntm.get_topic_distribution(train_feature, 'inference').detach().to(
        'cpu').numpy()
    test_representation = model.ntm.get_topic_distribution(test_feature, 'inference').detach().to(
        'cpu').numpy()
    classify_model.fit(train_representation, train_dataset[2])
    prediction = classify_model.predict_proba(test_representation)
    performance = evaluation(prediction, test_dataset[2])
    return performance, classify_model


def write_model_info(classify_model, representation_model, word_index_map):
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
        assert len(index_word_list) == len(single_topic_word_distribution)-1
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
    with open(topic_word_distribution_csv.format(timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(topic_word_write)

    # 数据
    classify_weight_write = [['' for _ in range(len(classify_weight)+4)] for _ in range(len(classify_weight[0])+4)]
    for i in range(len(classify_weight)):
        classify_weight_write[0][i+1] = str(i)
    for key in diagnosis_map:
        classify_weight_write[diagnosis_map[key]+1][0] = key
    for i in range(len(classify_weight)):
        for j in range(len(classify_weight[i])):
            classify_weight_write[j+1][i+1] = classify_weight[i, j]
    with open(classify_weight_csv.format(timestamp), 'w', encoding='utf-8-sig', newline='') as f:
        csv.writer(f).writerows(classify_weight_write)

    print('')


def main(args_):
    batch_size = args_['batch_size']
    hidden_size_ntm = args_['hidden_size_ntm']
    topic_number_ntm = args_['topic_number_ntm']
    learning_rate = args_['learning_rate']
    vocab_size_ntm = args_['vocab_size_ntm']
    epoch_number = args_['epoch_number']
    contrastive_coefficient = args_['contrastive_coefficient']
    similarity_coefficient = args_['similarity_coefficient']
    topic_coefficient = args_['topic_coefficient']
    ntm_coefficient = args_['ntm_coefficient']
    device = args_['device']
    model = args_['model']
    tau = args_['tau']
    diagnosis_size = args_['diagnosis_size']
    dataset_name = args_['dataset_name']
    read_from_cache = args_['read_from_cache']
    # read_from_cache = False
    for key in args_:
        logger.info('{}: {}'.format(key, args_[key]))

    # print('topic number: {}, epoch number: {}, vocab size: {}'.
    #       format(topic_number_ntm, epoch_number, vocab_size_ntm))
    # device = 'cuda:4'
    # for contrastive_coefficient in (0, 0, 0, 0, 0, 0):
    train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number, topic_coefficient,
          contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau, model, dataset_name,
          diagnosis_size, read_from_cache)


if __name__ == '__main__':
    main(args)
