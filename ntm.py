from torch import nn
from torch.nn import Linear, Sequential, Tanh
from torch.nn import init
import torch


class RevisedNTM(nn.Module):
    def __init__(self, hidden_size, topic_size, vocab_size, device):
        super(RevisedNTM, self).__init__()
        self.hidden = Sequential(
            Linear(vocab_size, hidden_size, bias=False),
            nn.LeakyReLU()
        )
        init.xavier_normal_(self.hidden[0].weight)
        self.normal = NormalParameter(hidden_size, topic_size, device)
        self.topics = RevisedNTMTopics(topic_size, vocab_size, device)
        self.h_to_z = nn.Softmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        h_1 = self.hidden(x)
        mu, log_sigma = self.normal(h_1)
        kld = kld_normal(mu, log_sigma)

        h_2 = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
        z = self.h_to_z(h_2)
        # print(torch.sum(z, dim=1))
        log_prob = self.topics(z)
        rec_loss = - (log_prob * x).sum(dim=-1)

        minus_elbo = rec_loss + kld
        return {
            'ntm_loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld,
            'h': h_2
        }

    def get_topics(self):
        return self.topics.get_topics()

    def get_topic_distribution(self, x, stage):
        h_1 = self.hidden(x)
        mu, log_sigma = self.normal(h_1)
        # hidden_parameter_numpy = self.hidden[0].weight.data.detach().to('cpu').numpy()
        # observe_mu = mu.detach().to('cpu').numpy()
        # h_1_numpy = h_1.detach().to('cpu').numpy()
        # observe_sigma = torch.exp(log_sigma).detach().to('cpu').numpy()
        if stage == 'inference':
            h_2 = mu
        else:
            h_2 = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
        #
        # mu_numpy = h_2.detach().to('cpu').numpy()
        # mu_weight_numpy = self.normal.mu.weight.data.detach().to('cpu').numpy()
        # sigma_weight_numpy = self.normal.log_sigma.weight.data.detach().to('cpu').numpy()
        prob = self.softmax(h_2)
        # torch.sum(prob, dim=1)
        # prob_numpy = prob.detach().to('cpu').numpy()
        return prob


class RevisedNTMTopics(nn.Module):
    def __init__(self, topic_size, vocab_size, device, bias=False):
        super(RevisedNTMTopics, self).__init__()
        self.topic_size = topic_size
        self.device = device
        self.vocab_size = vocab_size
        self.topic = nn.Linear(vocab_size, topic_size, bias=bias)
        self.normalize = nn.Softmax(dim=1)
        init.xavier_normal_(self.topic.weight)

    def forward(self, z):
        # return the log_prob of vocab distribution
        # print(torch.sum(topic_word_distribution, dim=1))
        topic = self.topic(torch.eye(self.vocab_size).to(self.device)).transpose(0, 1)
        topic_normalize = self.normalize(topic)
        # topic_normalize = self.topic_normalize_detach(topic)
        mixture_distribution = torch.matmul(z, topic_normalize)
        # print(torch.sum(mixture_distribution, dim=1))
        log_mixture_distribution = torch.log(mixture_distribution)
        return log_mixture_distribution

    # def topic_normalize_detach(self, topic):
    #     topic = topic - torch.max(topic.detach(), dim=1, keepdim=True).values
    #     normalize = (torch.exp(topic) / torch.sum(torch.exp(topic.detach()), dim=1, keepdim=True)).to(self.device)
    #     return normalize

    def get_topics(self):
        topic_distribution = torch.softmax(self.topic.weight.data, dim=1)
        return topic_distribution

    def get_topic_word_logit(self):
        """topic x V.
        Return the logit instead of probability distribution
        """
        return self.topic.weight


class NTM(nn.Module):
    def __init__(self, hidden_size, topic_size, vocab_size, device):
        super(NTM, self).__init__()
        self.hidden = Sequential(
            Linear(vocab_size, hidden_size),
            nn.LeakyReLU()
        )
        init.xavier_normal_(self.hidden[0].weight)
        init.normal_(self.hidden[0].bias)

        self.normal = NormalParameter(hidden_size, topic_size, device)
        self.h_to_z = Identity()
        self.topics = NTMTopics(topic_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)
        self.device = device

    def forward(self, x):
        h_1 = self.hidden(x)
        mu, log_sigma = self.normal(h_1)
        kld = kld_normal(mu, log_sigma)

        h_2 = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
        z = self.h_to_z(h_2)
        log_prob = self.topics(z)
        rec_loss = - (log_prob * x).sum(dim=-1)

        minus_elbo = rec_loss + kld
        return {
            'ntm_loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld,
            'h': h_2
        }

    def get_topics(self):
        return self.topics.get_topics()

    def get_topic_distribution(self, x, stage):
        h = self.hidden(x)
        mu, log_sigma = self.normal(h)
        # observe_mu = mu.detach().to('cpu').numpy()
        # observe_sigma = torch.exp(log_sigma).detach().to('cpu').numpy()
        if stage == 'inference':
            z = mu
        else:
            z = torch.zeros_like(mu).normal_() * torch.exp(log_sigma) + mu
        prob = self.softmax(z)
        return prob


def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


def topic_covariance_penalty(topic_emb, eps=1e-12):
    """topic_emb: T x topic_dim."""
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + eps)
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    mean = cosine.mean()
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean


def topic_embedding_weighted_penalty(embedding_weight, topic_word_logit, eps=1e-12):
    """embedding_weight: V x dim, topic_word_logit: T x V."""
    w = topic_word_logit.transpose(0, 1)  # V x T

    nv = embedding_weight / (torch.norm(embedding_weight, dim=1, keepdim=True) + eps)  # V x dim
    nw = w / (torch.norm(w, dim=0, keepdim=True) + eps)  # V x T
    t = nv.transpose(0, 1) @ w  # dim x T
    nt = t / (torch.norm(t, dim=0, keepdim=True) + eps)  # dim x T
    s = nv @ nt  # V x T
    return -(s * nw).sum()  # minus for minimization


class NTMTopics(nn.Module):
    def __init__(self, k, vocab_size, bias=False):
        super(NTMTopics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)
        init.xavier_normal_(self.topic.weight)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logit instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


class NormalParameter(nn.Module):
    def __init__(self, in_features, out_features, device):
        super(NormalParameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.device = device
        self.reset_parameters()

    def forward(self, h):
        return self.mu(h), self.log_sigma(h)

    def reset_parameters(self):
        init.xavier_normal_(self.mu.weight)
        init.xavier_normal_(self.log_sigma.weight)
        init.normal_(self.mu.bias)
        init.normal_(self.log_sigma.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(*input_):
        if len(input_) == 1:
            return input_[0]
        return input_
