import numpy as np
from sklearn.neural_network import MLPClassifier
from util import five_fold_datasets, dataset_format, evaluation, data_embedding
from config import tokenize_data_save_path
import random


def train(data):
    accuracy_list = []
    for i in range(5):
        print('iter: {}'.format(i))
        test_dataset, train_dataset = data[i], []
        for j in range(5):
            if i != j:
                for item in data[j]:
                    train_dataset.append(item)
        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(),
            max_iter=2000
        )
        mlp_model.fit(train_dataset[0], train_dataset[1])
        prediction = mlp_model.predict_proba(test_dataset[0])
        accuracy = evaluation(prediction, test_dataset[1])
        print('iter {}, accuracy: {}'.format(i, accuracy))
        accuracy_list.append(accuracy)
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    embedding = data_embedding(tokenize_data_save_path, True)
    index_list = [i for i in range(len(embedding))]
    random.shuffle(index_list)
    five_fold_data = five_fold_datasets(embedding, index_list)
    train(five_fold_data)


if __name__ == '__main__':
    main()
