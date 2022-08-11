import numpy as np
import os
import csv
from config import args
from config import cache_dir, logger
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
from util import evaluation, dataset_selection, dataset_format
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset, load_metric
from torch import nn
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
# from torch.nn import init


def calculate_performance(test_dataloader, model):
    predict_list, label_list = [], []
    softmax = nn.Softmax()
    for batch_data in test_dataloader:
        _, _, label, token, mask, _ = batch_data
        output = model(token, mask).detach()
        predict = list(softmax(output).to('cpu').numpy())
        label = list(label.detach().to('cpu').numpy())
        predict_list += predict
        label_list += label
    performance = evaluation(predict_list, label_list)
    return performance


def process_dataset(dataset_name, cut_length, vocab_size, diagnosis_size, read_from_cache, test_set_num):
    folder_path = os.path.abspath('./data/data_utf_8/pnnlm_{}_{}'.format(dataset_name, test_set_num))
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size, diagnosis_size,
                                                       cut_length, read_from_cache)
    assert test_set_num in {0, 1, 2, 3, 4}
    csv_train = [["identifier", "label", "text"]]
    csv_test = [["identifier", "label", "text"]]
    for i in range(5):
        for item in five_fold_data[i]:
            identifier, _, _, disease_index, emr = item
            if i == test_set_num:
                csv_test += [[identifier, disease_index, emr]]
            else:
                csv_train += [[identifier, disease_index, emr]]

    csv_train_path = os.path.join(folder_path, 'train.csv')
    csv_test_path = os.path.join(folder_path, 'test.csv')
    csv.writer(open(csv_train_path, 'w', encoding='utf-8-sig', newline='')).writerows(csv_train)
    csv.writer(open(csv_test_path, 'w', encoding='utf-8-sig', newline='')).writerows(csv_test)
    return csv_train_path, csv_test_path


def no_finetune(data, datatype):
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
            max_iter=200
        )
        if datatype == 'raw':
            mlp_model.fit(train_dataset[0], train_dataset[2])
            prediction = mlp_model.predict_proba(test_dataset[0])
            performance = evaluation(prediction, test_dataset[2])
        else:
            mlp_model.fit(train_dataset[1], train_dataset[2])
            prediction = mlp_model.predict_proba(test_dataset[1])
            performance = evaluation(prediction, test_dataset[2])
        print('iter {}, performance: {}'.format(i, performance))
        accuracy_list.append(performance['accuracy'])
    print('accuracy: {}'.format(np.average(accuracy_list)))


def finetune(dataset_name, cut_length, vocab_size, diagnosis_size, read_from_cache, test_set_num, max_iteration):
    train_path, test_path = \
        process_dataset(dataset_name, cut_length, vocab_size, diagnosis_size, read_from_cache, test_set_num)
    data = load_dataset('csv', data_files={'train': train_path, 'test': test_path})
    # test_data = load_dataset('csv', data_files=test_path)
    if dataset_name == 'hzsph':
        tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-macbert-base', cache_dir=cache_dir)
        tokenizer.model_max_length = 512
        tokenizer.max_len_single_sentence = 510
        model = AutoModelForSequenceClassification.\
            from_pretrained('hfl/chinese-macbert-base', num_labels=3, cache_dir=cache_dir)
        per_device_batch_size = 4
        # num_train_epochs = 5
    elif dataset_name == 'mimic-iii':
        tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096", cache_dir=cache_dir)
        model = AutoModelForSequenceClassification. \
            from_pretrained("allenai/longformer-base-4096", num_labels=10, cache_dir=cache_dir)
        per_device_batch_size = 2
        # num_train_epochs = 2
    else:
        raise ValueError('')

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, padding=True)

    tokenized_data = data.map(preprocess_function, batched=True)
    # tokenized_test = test_data.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./data/pnnlm_result/",
        learning_rate=1e-5,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        weight_decay=0.01
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data['train'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.create_optimizer()

    num_iter, epoch = 0, 0
    while True:
        data_loader = trainer.get_train_dataloader()
        for batch in tqdm(data_loader):
            trainer.optimizer.zero_grad()
            trainer.training_step(trainer.model, batch)
            trainer.optimizer.step()

            if num_iter % 50 == 0:
                pred = trainer.predict(tokenized_data['test']).predictions
                pred_prob = np.exp(pred - np.max(pred, axis=1, keepdims=True)) / \
                    np.sum(np.exp(pred - np.max(pred, axis=1, keepdims=True)), axis=1, keepdims=True)
                performance = evaluation(pred_prob, tokenized_data['test']['label'])
                logger.info('dataset: {}, iter: {}, test_num: {}'.format(dataset_name, num_iter, test_set_num))
                logger.info(performance)

            num_iter += 1
        logger.info('epoch {} finished'.format(epoch))
        epoch += 1
        if num_iter >= max_iteration:
            break
    logger.info('finished')
    # print('accuracy: {}'.format(np.average(accuracy_list)))


def main():
    # dataset_name = args['dataset_name']
    diagnosis_size = args['diagnosis_size']
    vocab_size_ntm = args['vocab_size_ntm']
    read_from_cache = args['read_from_cache']
    cut_length = args['cut_length']
    test_set_num = args['test_set_num']
    dataset_name = 'hzsph'
    max_iteration = 5000

    # datatype = 'raw'
    # print('dataset name: {}, datatype: {}'.format(dataset_name, datatype))
    # five_fold_data, word_index_map = dataset_selection(dataset_name, vocab_size_ntm, diagnosis_size,
    #                                                    cut_length, read_from_cache)
    # no_finetune(five_fold_data, datatype)
    logger.info('start finetune')
    for test_set_num in {0, 1, 2, 3, 4}:
        for dataset_name in 'hzsph', 'mimic-iii':
            finetune(dataset_name, cut_length, vocab_size_ntm, diagnosis_size, read_from_cache, test_set_num,
                     max_iteration)


if __name__ == '__main__':
    main()
