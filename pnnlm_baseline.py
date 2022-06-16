import numpy as np
from sklearn.neural_network import MLPClassifier
from util import evaluation, dataset_selection, dataset_format


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
        mlp_model.fit(train_dataset[1], train_dataset[2])
        prediction = mlp_model.predict_proba(test_dataset[1])
        performance = evaluation(prediction, test_dataset[2])
        print('iter {}, performance: {}'.format(i, performance))
        accuracy_list.append(performance['accuracy'])
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    vocab_size = 10000
    diagnosis_size = 50
    read_from_cache = True
    dataset_name = 'mimic-iii'  # mimic-iii hzsph
    print('dataset name: {}'.format(dataset_name))
    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size, read_from_cache)
    train(five_fold_data)


if __name__ == '__main__':
    main()
