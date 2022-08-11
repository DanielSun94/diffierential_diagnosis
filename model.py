import torch
from torch import diag, ones
from torch.nn import Module
from ntm import NTM, RevisedNTM
import numpy as np
from torch.utils.data import Dataset


class TopicAwareNTM(Module):
    def __init__(self, hidden_size, topic_number, vocab_size, device, tau, ntm):
        super(TopicAwareNTM, self).__init__()
        self.device = device
        if ntm == 'ntm':
            self.ntm = NTM(hidden_size, topic_number, vocab_size, device).to(device)
        elif ntm == 'rntm':
            self.ntm = RevisedNTM(hidden_size, topic_number, vocab_size, device).to(device)
        else:
            raise ValueError('')
        self.tau = tau
        self.topic_number = topic_number

    def forward(self, origin_feature, nnlm_representation, same_class_mat):
        # ntm loss
        origin_feature = origin_feature.to(self.device)
        nnlm_representation = nnlm_representation.to(self.device)
        same_class_mat = same_class_mat.to(self.device)

        output = self.ntm(origin_feature)
        ntm_loss, h = output['ntm_loss'], output['h']

        # topic similarity loss
        topic_word_distribution = self.ntm.topics.get_topics()
        # print(torch.sum(topic_word_distribution, dim=1))
        topic_number = self.topic_number
        topic_word_similarity = self.pairwise_similarity(topic_word_distribution, topic_word_distribution)
        multiply_mat = (ones([topic_number, topic_number])-diag(ones([topic_number]))).to(self.device)
        topic_word_similarity = torch.multiply(topic_word_similarity, multiply_mat)
        output['topic_word_loss'] = topic_word_similarity
        # output['topic_word_loss'] = torch.FloatTensor(0)

        # similarity loss
        doc_topic_distribution = torch.softmax(h, dim=1)
        doc_topic_similarity = self.pairwise_similarity(doc_topic_distribution, doc_topic_distribution)
        semantic_similarity = self.pairwise_similarity(nnlm_representation, nnlm_representation)
        difference = (semantic_similarity-doc_topic_similarity)
        similarity_loss = torch.sum(difference*difference)
        output['similarity_loss'] = similarity_loss
        # output['similarity_loss'] = torch.FloatTensor(0)

        # contrastive loss, info NCE
        size = doc_topic_similarity.shape[0]
        contrastive_info_mat = torch.multiply(doc_topic_similarity, (ones([size, size]) -
                                                                     diag(ones(size))).to(self.device))
        same_class_similarity = contrastive_info_mat * same_class_mat
        different_class_similarity = contrastive_info_mat * (1-same_class_mat)
        contrastive_loss = different_class_similarity - same_class_similarity
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


class EHRDataset(Dataset):
    def __init__(self, dataset):
        self.feature = dataset[0]
        self.representation = dataset[1]
        self.label = dataset[2]
        self.string = dataset[3]

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        return self.feature[idx], self.representation[idx], self.label[idx], self.string[idx]


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
    label = torch.LongTensor(np.array([item[2] for item in batch]))
    string = [item[3] for item in batch]
    return feature, representation, label, string, same_class_index
