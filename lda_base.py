import numpy as np
from util import five_fold_datasets, dataset_format, evaluation, bag_of_word_reorganize
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation


def train(topic_number, vocab_size):
    word_index_map, reformat_data = bag_of_word_reorganize(vocab_size)
    five_fold_data = five_fold_datasets(reformat_data)
    accuracy_list = []
    for i in range(5):
        # print('iter: {}'.format(i))
        test_dataset, train_dataset = five_fold_data[i], []
        for j in range(5):
            if i != j:
                for item in five_fold_data[j]:
                    train_dataset.append(item)
        train_dataset, test_dataset = dataset_format(train_dataset), dataset_format(test_dataset)
        lda = LatentDirichletAllocation(n_components=topic_number, random_state=0)
        lda.fit(train_dataset[0])
        train_representation = lda.transform(train_dataset[0])
        test_representation = lda.transform(test_dataset[0])
        mlp_model = MLPClassifier(hidden_layer_sizes=(), max_iter=2000)
        mlp_model.fit(train_representation, train_dataset[1])
        prediction = mlp_model.predict_proba(test_representation)
        accuracy = evaluation(prediction, test_dataset[1])
        # print('iter {}, accuracy: {}'.format(i, accuracy))
        accuracy_list.append(accuracy)
    print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    for topic_num in 5, 10, 30, 20:
        for vocab_size in 1000, 3000, 5000, 7000, 9000:
            print('topic number: {}, vocab_number: {}'.format(topic_num, vocab_size))
            train(topic_num, vocab_size)


if __name__ == '__main__':
    main()
