import torch
from torch import nn
from torch.nn import Linear, Sequential, Tanh
from torch.nn import init
import numpy as np
from util import bag_of_word_reorganize, five_fold_datasets, dataset_format, evaluation
from config import hidden_size_ntm, device
import torch.optim as optim
from sklearn.neural_network import MLPClassifier


def main():
    n_iter = 2000
    for topic_number_ntm in 50, 100, 200, 300, 400:
        for vocab_size_ntm in 10000, 20000:
            word_index_map, reformat_data = bag_of_word_reorganize(vocab_size_ntm)
            five_fold_data = five_fold_datasets(reformat_data)
            accuracy_list = list()
            for i in range(5):
                # print('iter: {}'.format(i))
                test_dataset, train_dataset = five_fold_data[i], []
                for j in range(5):
                    if i != j:
                        for item in five_fold_data[j]:
                            train_dataset.append(item)
                train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
                train_feature = torch.FloatTensor(train_dataset[0]).to(device)
                test_feature = torch.FloatTensor(test_dataset[0]).to(device)
                model = NTM(hidden_size_ntm, topic_number_ntm, vocab_size_ntm).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for _ in range(0, n_iter):
                    optimizer.zero_grad()
                    output = model(train_feature)
                    # print('loss: {}'.format(output['loss'].mean()))
                    output['loss'].mean().backward()
                    optimizer.step()

                train_representation = model.get_topic_distribution(train_feature).detach().to('cpu').numpy()
                test_representation = model.get_topic_distribution(test_feature).detach().to('cpu').numpy()
                mlp_model = MLPClassifier(hidden_layer_sizes=(), max_iter=2000)
                mlp_model.fit(train_representation, train_dataset[1])
                prediction = mlp_model.predict_proba(test_representation)
                accuracy = evaluation(prediction, test_dataset[1])
                # print('iter {}, accuracy: {}'.format(i, accuracy))
                accuracy_list.append(accuracy)
            print('topic number: {}, vocab size: {}'.format(topic_number_ntm, vocab_size_ntm))
            print('accuracy: {}'.format(np.average(accuracy_list)))


class NTM(nn.Module):
    def __init__(self, hidden_size, topic_size, vocab_size):
        super(NTM, self).__init__()
        self.hidden = Sequential(
            Linear(vocab_size, hidden_size),
            Tanh()
        )
        self.normal = NormalParameter(hidden_size, topic_size)
        self.h_to_z = Identity()
        self.topics = Topics(topic_size, vocab_size)

    def forward(self, x, n_sample=1):
        h = self.hidden(x)
        mu, log_sigma = self.normal(h)

        kld = kld_normal(mu, log_sigma)
        rec_loss = 0
        for i in range(n_sample):
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            z = self.h_to_z(z)
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        rec_loss = rec_loss / n_sample

        minus_elbo = rec_loss + kld

        return {
            'loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld
        }

    def get_topics(self):
        return self.topics.get_topics()

    def get_topic_distribution(self, x, n_sample=1000):
        softmax = nn.Softmax(dim=1)
        h = self.hidden(x)
        mu, log_sigma = self.normal(h)
        prob = torch.zeros_like(mu)
        for i in range(n_sample):
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
            prob += softmax(z)
        prob = prob / n_sample
        return prob


def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)



def topic_covariance_penalty(topic_emb, EPS=1e-12):
    """topic_emb: T x topic_dim."""
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + EPS)
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    mean = cosine.mean()
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean


def topic_embedding_weighted_penalty(embedding_weight, topic_word_logit, EPS=1e-12):
    """embedding_weight: V x dim, topic_word_logit: T x V."""
    w = topic_word_logit.transpose(0, 1)  # V x T

    nv = embedding_weight / (torch.norm(embedding_weight, dim=1, keepdim=True) + EPS)  # V x dim
    nw = w / (torch.norm(w, dim=0, keepdim=True) + EPS)  # V x T
    t = nv.transpose(0, 1) @ w  # dim x T
    nt = t / (torch.norm(t, dim=0, keepdim=True) + EPS)  # dim x T
    s = nv @ nt  # V x T
    return -(s * nw).sum()  # minus for minimization


class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class NormalParameter(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalParameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def forward(self, h):
        return self.mu(h), self.log_sigma(h)

    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


def test():
    hidden_size = 500
    topic_size = 50
    vocab_size = 1994
    model = NTM(hidden_size=hidden_size, topic_size=topic_size, vocab_size=vocab_size)
    x = np.random.randint(0, 100, [64, 1994])
    model(torch.FloatTensor(x))
    print('accomplish')


if __name__ == '__main__':
    # test()
    main()
