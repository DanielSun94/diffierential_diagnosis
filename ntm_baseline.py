import torch
from ntm import NTM
import numpy as np
from util import bag_of_word_reorganize, five_fold_datasets, dataset_format, evaluation
from config import hidden_size_ntm, device
import torch.optim as optim
import random
from sklearn.neural_network import MLPClassifier


def main():
    n_iter = 2000
    for topic_number_ntm in 100, :
        for vocab_size_ntm in 10000, :
            word_index_map, reformat_data = bag_of_word_reorganize(vocab_size_ntm)
            index_list = [i for i in range(len(reformat_data))]
            random.shuffle(index_list)
            five_fold_data = five_fold_datasets(reformat_data, index_list)
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
                    output['ntm_loss'].mean().backward()
                    optimizer.step()

                train_representation = model.get_topic_distribution(train_feature).detach().to('cpu').numpy()
                test_representation = model.get_topic_distribution(test_feature).detach().to('cpu').numpy()
                mlp_model = MLPClassifier(hidden_layer_sizes=(), max_iter=6000)
                mlp_model.fit(train_representation, train_dataset[1])
                prediction = mlp_model.predict_proba(test_representation)
                accuracy = evaluation(prediction, test_dataset[1])
                # print('iter {}, accuracy: {}'.format(i, accuracy))
                accuracy_list.append(accuracy)
            print('topic number: {}, vocab size: {}'.format(topic_number_ntm, vocab_size_ntm))
            print('accuracy: {}'.format(np.average(accuracy_list)))


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
