import numpy as np
from util import evaluation, dataset_selection, dataset_format
from config import args
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation


def train(topic_number, vocab_size, dataset_name, diagnosis_size, read_from_cache):
    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size, read_from_cache)
    accuracy_list = []
    for i in range(5):
        print('iter: {}'.format(i))
        test_dataset, train_dataset = five_fold_data[i], []
        for j in range(5):
            if i != j:
                for item in five_fold_data[j]:
                    train_dataset.append(item)
        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        lda = LatentDirichletAllocation(n_components=topic_number)
        lda.fit(train_dataset[0])
        train_representation = lda.transform(train_dataset[0])
        test_representation = lda.transform(test_dataset[0])
        mlp_model = MLPClassifier(hidden_layer_sizes=(), max_iter=2000)
        mlp_model.fit(train_representation, train_dataset[2])
        prediction = mlp_model.predict_proba(test_representation)
        performance = evaluation(prediction, test_dataset[2])
        print('iter {}, performance: {}'.format(i, performance))
        accuracy_list.append(performance['accuracy'])
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    diagnosis_size = args['diagnosis_size']
    vocab_size_ntm = args['vocab_size_ntm']
    read_from_cache = args['read_from_cache']
    dataset_name = 'hzsph'
    print('dataset name: {}'.format(dataset_name))
    for topic_num in 10, :
        for vocab_size in vocab_size_ntm, :
            print('topic number: {}, vocab_number: {}'.format(topic_num, vocab_size))
            train(topic_num, vocab_size, dataset_name, diagnosis_size, read_from_cache)


if __name__ == '__main__':
    main()
