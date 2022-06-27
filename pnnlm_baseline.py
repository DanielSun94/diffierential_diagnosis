import numpy as np
from config import args
from sklearn.neural_network import MLPClassifier
from util import evaluation, dataset_selection, dataset_format


def train(data, datatype):
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
        if datatype == 'raw':
            mlp_model.fit(train_dataset[1], train_dataset[2])
            prediction = mlp_model.predict_proba(test_dataset[1])
            performance = evaluation(prediction, test_dataset[2])
        else:
            mlp_model.fit(train_dataset[1], train_dataset[2])
            prediction = mlp_model.predict_proba(test_dataset[1])
            performance = evaluation(prediction, test_dataset[2])
        print('iter {}, performance: {}'.format(i, performance))
        accuracy_list.append(performance['accuracy'])
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    diagnosis_size = args['diagnosis_size']
    vocab_size_ntm = args['vocab_size_ntm']
    read_from_cache = args['read_from_cache']
    dataset_name = args['dataset_name']
    dataset_name = 'mimic-iii'
    read_from_cache = False
    datatype = 'raw'
    print('dataset name: {}'.format(dataset_name))
    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size_ntm, diagnosis_size, read_from_cache)
    train(five_fold_data, datatype)


if __name__ == '__main__':
    main()
