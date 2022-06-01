import torch
from torch import diag, ones
from torch.nn import Module
from ntm import NTM
from torch import optim
import numpy as np
import os
import pickle
import random
from config import tokenize_data_save_path, contrastive_ntm_data_cache, args, logger
from util import five_fold_datasets, bag_of_word_reorganize, dataset_format, evaluation, data_embedding
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, Dataset


class ContrastiveNTM(Module):
    def __init__(self, hidden_size, topic_number, vocab_size, device, tau):
        super(ContrastiveNTM, self).__init__()
        self.device = device
        self.ntm = NTM(hidden_size, topic_number, vocab_size, device).to(device)
        self.tau = tau
        self.topic_number = topic_number

    def forward(self, origin_feature, nnlm_representation, same_class_mat, sample_size):
        # ntm loss
        origin_feature = origin_feature.to(self.device)
        nnlm_representation = nnlm_representation.to(self.device)
        same_class_mat = same_class_mat.to(self.device)

        output = self.ntm(origin_feature, sample_size)
        ntm_loss, z = output['ntm_loss'], output['z']

        # topic similarity loss
        topic_word_distribution = self.ntm.topics.get_topics()
        # print(torch.sum(topic_word_distribution, dim=1))
        topic_number = self.topic_number
        topic_word_similarity = self.pairwise_similarity(topic_word_distribution, topic_word_distribution)
        multiply_mat = (ones([topic_number, topic_number])-diag(ones([topic_number]))).to(self.device)
        topic_word_similarity = torch.multiply(topic_word_similarity, multiply_mat)
        output['topic_word_loss'] = -1 * topic_word_similarity
        # output['topic_word_loss'] = torch.FloatTensor(0)

        # similarity loss
        doc_topic_distribution = torch.softmax(z, dim=1)
        doc_topic_similarity = self.pairwise_similarity(doc_topic_distribution, doc_topic_distribution)
        semantic_similarity = self.pairwise_similarity(nnlm_representation, nnlm_representation)
        difference = (semantic_similarity-doc_topic_similarity)
        similarity_loss = torch.sum(difference*difference)
        output['similarity_loss'] = similarity_loss
        # output['similarity_loss'] = torch.FloatTensor(0)

        # contrastive loss, info NCE
        # 不算自身的相似度
        size = doc_topic_similarity.shape[0]
        contrastive_info_mat = torch.multiply(doc_topic_similarity, (ones([size, size]) -
                                                                     diag(ones(size))).to(self.device))
        contrastive_info_mat = torch.exp(contrastive_info_mat)

        nominator = contrastive_info_mat * same_class_mat
        nominator = torch.sum(nominator, dim=1)
        denominator = torch.sum(contrastive_info_mat, dim=1)
        contrastive_loss = torch.tensor(-1) * torch.log(nominator / denominator)
        contrastive_loss = torch.sum(contrastive_loss)
        output['contrastive_loss'] = contrastive_loss
        # output['contrastive_loss'] = torch.FloatTensor(0)
        return output

    def pairwise_similarity(self, mat_1, mat_2, eps=10e-8):
        mat_2 = torch.transpose(mat_2, 0, 1)
        similarity = torch.matmul(mat_1, mat_2)
        diagonal = torch.diag(torch.diag(similarity.detach())).to(self.device)
        diagonal = diagonal + torch.FloatTensor(torch.eye(len(mat_1)) * eps).to(self.device)
        sqrt_diagonal_inv = torch.sqrt(torch.inverse(diagonal))
        similarity = torch.matmul(similarity, sqrt_diagonal_inv)
        similarity = torch.matmul(torch.transpose(similarity, 0, 1), sqrt_diagonal_inv)
        return similarity


def load_data(read_from_cache, vocab_size_ntm):
    if read_from_cache and os.path.exists(contrastive_ntm_data_cache):
        data = pickle.load(open(contrastive_ntm_data_cache, 'rb'))
    else:
        word_index_map, reformat_data = bag_of_word_reorganize(vocab_size_ntm)
        embedding = data_embedding(tokenize_data_save_path, overwrite=False)
        index_list = [i for i in range(len(reformat_data))]
        random.shuffle(index_list)
        five_fold_feature = five_fold_datasets(reformat_data, index_list)
        five_fold_embedding = five_fold_datasets(embedding, index_list)
        five_fold_data = list()
        for fold_feature, fold_embedding in zip(five_fold_feature, five_fold_embedding):
            fold_info = list()
            for item_feature, item_embedding in zip(fold_feature, fold_embedding):
                assert item_feature[0] == item_embedding[0]
                key, feature, diagnosis, embedding = \
                    item_feature[0], item_feature[1], item_feature[2], item_embedding[1]
                fold_info.append([key, feature, embedding, diagnosis])
            five_fold_data.append(fold_info)
        pickle.dump(five_fold_data, open(contrastive_ntm_data_cache, 'wb'))
        data = five_fold_data
    logger.info('data loaded')
    return data


def train(batch_size, hidden_size, topic_number, learning_rate, vocab_size, epoch_number, topic_coefficient,
          contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau, read_from_cache):
    data = load_data(read_from_cache, vocab_size)
    performance_list = list()
    sample_size = args['sample_size']
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
        model = ContrastiveNTM(hidden_size, topic_number, vocab_size, device, tau)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(epoch_number):
            epoch_loss = list()
            for batch_data in train_dataloader:
                feature, representation, _, same_class_mat = batch_data
                optimizer.zero_grad()
                output = model(feature, representation, same_class_mat, sample_size)

                contrastive_loss = output['contrastive_loss'].mean()
                similarity_loss = output['similarity_loss'].mean()
                ntm_loss = output['ntm_loss'].mean()
                topic_word_loss = output['topic_word_loss'].mean()
                # loss_sum = contrastive_loss.detach() + similarity_loss.detach() + ntm_loss.detach()
                # loss = contrastive_loss * loss_sum * contrastive_coefficient / contrastive_loss.detach() + \
                #     similarity_loss * loss_sum * similarity_coefficient / similarity_loss.detach() + \
                #     ntm_loss * loss_sum * ntm_coefficient / ntm_loss.detach()
                # print('loss: {}, ntm loss: {}, similarity loss: {}, contrastive loss: {}'.format(
                #     loss.item(), ntm_loss.item(), similarity_loss.item(), contrastive_loss.item()
                # ))
                # loss = ntm_loss
                loss = ntm_loss * ntm_coefficient + similarity_loss * similarity_coefficient + \
                    contrastive_loss * contrastive_coefficient + topic_word_loss * topic_coefficient
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.mean().detach().to('cpu').numpy())
            # logger.info('epoch: {}, average loss: {}'.format(epoch, np.average(epoch_loss)))

        if args['classify_model'] == 'knn':
            classify_model = KNeighborsClassifier(n_neighbors=3)
        elif args['classify_model'] == 'nn':
            classify_model = MLPClassifier(hidden_layer_sizes=(), max_iter=6000)
        elif args['classify_model'] == 'hnn':
            classify_model = MLPClassifier(hidden_layer_sizes=(10, ), max_iter=6000)
        else:
            raise ValueError('')
        train_feature = torch.FloatTensor(np.array(train_dataset[0])).to(device)
        test_feature = torch.FloatTensor(np.array(test_dataset[0])).to(device)
        train_representation = model.ntm.get_topic_distribution(train_feature, sample_size, 'inference')\
            .detach().to('cpu').numpy()
        test_representation = model.ntm.get_topic_distribution(test_feature, sample_size, 'inference')\
            .detach().to('cpu').numpy()
        classify_model.fit(train_representation, train_dataset[2])
        prediction = classify_model.predict_proba(test_representation)
        performance = evaluation(prediction, test_dataset[2])
        # print('iter {}, accuracy: {}'.format(i, accuracy))
        performance_list.append(performance)
        logger.info('fold: {}, accuracy: {}'.format(i+1, performance['accuracy']))
    for key in performance_list[0]:
        score_list = [item[key] for item in performance_list]
        logger.info('average {}: {}'.format(key, np.average(score_list)))


class EHRDataset(Dataset):
    def __init__(self, dataset):
        self.feature = dataset[0]
        self.representation = dataset[1]
        self.label = dataset[2]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.representation[idx], self.label[idx]


def collate_fn(batch):
    batch = sorted(batch, key=lambda x: x[2])
    same_class_mat = np.zeros([len(batch), len(batch)])
    previous_class_index, previous_index = batch[0][2], 0
    for i in range(len(batch)):
        if batch[i][2] != previous_class_index:
            same_class_mat[previous_index: i, previous_index: i] = 1
            previous_index = i
            previous_class_index = batch[i][2]
        if i == len(batch) - 1:
            same_class_mat[previous_index: len(batch), previous_index: len(batch)] = 1

    same_class_index = torch.IntTensor(same_class_mat)
    feature = torch.FloatTensor(np.array([item[0] for item in batch]))
    representation = torch.FloatTensor(np.array([item[1] for item in batch]))
    label = torch.FloatTensor(np.array([item[2] for item in batch]))
    return feature, representation, label, same_class_index


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
    tau = args_['tau']
    read_from_cache = args_['read_from_cache']

    for key in args_:
        logger.info('{}: {}'.format(key, args_[key]))

    # print('topic number: {}, epoch number: {}, vocab size: {}'.
    #       format(topic_number_ntm, epoch_number, vocab_size_ntm))
    train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number, topic_coefficient,
          contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau, read_from_cache)


if __name__ == '__main__':
    main(args)


