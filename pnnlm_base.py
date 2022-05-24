import os
import pickle
from transformers import BertModel, BertTokenizer
from data_reformat import load_data, reconstruct_data
from config import cn_CLS_token, tokenize_data_save_path, neural_network_admission_parse_list, \
    neural_network_first_emr_parse_list, cache_dir, device, diagnosis_map
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from util import five_fold_datasets, dataset_format, evaluation


def data_embedding(save_path, overwrite=False):
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-macbert-base', cache_dir=cache_dir)
    model = BertModel.from_pretrained('hfl/chinese-macbert-base', cache_dir=cache_dir).to(device)

    if os.path.exists(save_path) and (not overwrite):
        return pickle.load(open(save_path, 'rb'))

    admission_data_dict, first_emr_record_dict = load_data()
    data = reconstruct_data(admission_data_dict, first_emr_record_dict, neural_network_admission_parse_list,
                            neural_network_first_emr_parse_list)
    embedding_data = dict()
    length_list = []
    for key in data:
        info_str, diagnosis = data[key]
        token = tokenizer(cn_CLS_token + ' ' + info_str)['input_ids']
        # token_str = tokenizer.tokenize(cn_CLS_token + ' ' + info_str)
        length_list.append(len(token))
        if len(token) > 512:
            token = token[:512]
            print('{} input id len is larger than 512'.format(key))
        embedding_data[key] = model(torch.LongTensor([token]).to('cuda:2'))['last_hidden_state'][0][0].\
            detach().cpu().numpy(), diagnosis_map[diagnosis]
    pickle.dump(embedding_data, open(save_path, 'wb'))

    print('average len {}'.format(np.average(length_list)))
    print('max len {}'.format(np.max(length_list)))
    print('min len {}'.format(np.min(length_list)))
    return embedding_data


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
    five_fold_data = five_fold_datasets(embedding)
    train(five_fold_data)


if __name__ == '__main__':
    main()
