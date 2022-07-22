from config import logger, args
from util import evaluation, dataset_selection, dataset_format, write_tendency, write_model_info, write_perplexity, \
    write_representation
from model import TopicAwareNTM, EHRDataset, collate_fn
import torch
from torch.utils.data import DataLoader
from torch import optim
from sklearn.neural_network import MLPClassifier
import numpy as np


def one_cv_train(batch_size, hidden_size, topic_number, learning_rate, vocab_size, epoch_number, topic_coefficient,
                 contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau, model_name,
                 dataset_name, diagnosis_size, read_from_cache, write_model_flag, write_tendency_flag,
                 write_perplexity_flag, write_representation_flag, test_set_num, write_model, write_path):
    cv_para = {
        "batch_size": batch_size, "hidden_size": hidden_size, "topic_number": topic_number, "model_name": model_name,
        "learning_rate": learning_rate, "vocab_size": vocab_size, "epoch_number": epoch_number, "tau": tau,
        "topic_coefficient": topic_coefficient, "contrastive_coefficient": contrastive_coefficient, "device": device,
        "similarity_coefficient": similarity_coefficient,  "ntm_coefficient": ntm_coefficient,
        "dataset_name": dataset_name, "diagnosis_size": diagnosis_size,  "read_from_cache": read_from_cache,
        "write_model_flag": write_model_flag,  "write_tendency_flag": write_tendency_flag,
        "write_perplexity_flag": write_perplexity_flag, "write_representation_flag": write_representation_flag,
        'test_set_num': test_set_num
    }

    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size, read_from_cache)

    performance_list = list()
    tendency_list = list()

    for key in cv_para:
        logger.info('{}: {}'.format(key, cv_para[key]))
    for i in range(5):
        # print('iter: {}'.format(i))
        test_dataset, train_dataset = five_fold_data[i], []
        for j in range(5):
            if i != j:
                for item in five_fold_data[j]:
                    train_dataset.append(item)
        if test_set_num is not None:
            if test_set_num != i:
                continue

        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        training_data = EHRDataset(train_dataset)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        while True:
            # 部分情况下，本模型在参数初始化不佳时会产生一些问题，此时直接重新来一遍，直到不出错为止
            try:
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

                    # 如果要记变化趋势，只需要记一个fold的变化趋势即可
                    if write_tendency_flag and i == 0 and epoch % 5 == 0:
                        performance, _ = calculate_performance(train_dataset, test_dataset, model, device)
                        tendency_list.append([epoch, performance])
                        logger.info('epoch: {}, fold: {}, accuracy: {}'.format(epoch, i + 1, performance['accuracy']))
                        logger.info('epoch: {}, average loss: {}'.format(epoch, np.average(epoch_loss)))
                performance, classify_model = calculate_performance(train_dataset, test_dataset, model, device)
            except ValueError as exception:
                logger.info(exception)
                continue
            # print('iter {}, accuracy: {}'.format(i, accuracy))
            performance_list.append(performance)
            for key in performance:
                logger.info('fold: {}, {}: {}'.format(i + 1, key, performance[key]))

            # 输出模型参数，只需要记一个fold的参数即可
            if write_model_flag and i == 0 and dataset_name == 'hzsph':
                write_model_info(classify_model, model, word_index_map, cv_para)
            if write_tendency_flag and i == 0:
                write_tendency(tendency_list, cv_para)
            if write_perplexity_flag and i == 0:
                write_perplexity(model, test_dataset, cv_para)
            if write_representation_flag and i == 0:
                write_representation(model, train_dataset, test_dataset, cv_para)
            if write_model and i == 0:
                assert write_path is not None
                torch.save(model.state_dict(), write_path)
            break


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
    repeat_time = args_['repeat_time']
    read_from_cache = args_['read_from_cache']
    write_model_flag, write_tendency_flag, write_perplexity_flag, write_representation_flag = True, True, True, True
    # read_from_cache = False
    for key in args_:
        logger.info('{}: {}'.format(key, args_[key]))

    # print('topic number: {}, epoch number: {}, vocab size: {}'.
    #       format(topic_number_ntm, epoch_number, vocab_size_ntm))
    # device = 'cuda:4'
    # for contrastive_coefficient in (0, 0, 0, 0, 0, 0):
    for _ in range(repeat_time):
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, write_model_flag, write_tendency_flag,
                     write_perplexity_flag, write_representation_flag, None)


if __name__ == '__main__':
    main(args)
