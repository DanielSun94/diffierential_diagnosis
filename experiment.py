from train import one_cv_train
from config import args, logger


def main():
    batch_size, hidden_size_ntm, model, tau, diagnosis_size, topic_number_ntm, learning_rate, vocab_size_ntm, \
        epoch_number, contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient, device, \
        dataset_name, repeat_time, read_from_cache, test_set_num, experiment_type, cut_length = \
        args['batch_size'], args['hidden_size_ntm'], args['model'], args['tau'], args['diagnosis_size'], \
        args['topic_number_ntm'], args['learning_rate'], args['vocab_size_ntm'],  args['epoch_number'], \
        args['contrastive_coefficient'], args['similarity_coefficient'], args['topic_coefficient'], \
        args['ntm_coefficient'], args['device'], args['dataset_name'], args['repeat_time'], args['read_from_cache'],\
        args['test_set_num'], args['experiment_type'], args['cut_length']

    for key in args:
        logger.info('{}: {}'.format(key, args[key]))

    if experiment_type == 'hyperparameter':
        for _ in range(args['repeat_time']):
            hyperparameter_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                    epoch_number, device, tau, model, diagnosis_size, read_from_cache, test_set_num,
                                    cut_length)
    else:
        output_representation(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                              epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                              ntm_coefficient, device, tau, model, diagnosis_size, read_from_cache, test_set_num)

        for topic_number_ntm in 3, 5, 8, 10, 15, 20, 30:
            perplexity_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                                ntm_coefficient, device, tau, model, diagnosis_size, read_from_cache,
                                test_set_num)

        convergence_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                             epoch_number, device, tau, model, diagnosis_size, read_from_cache, test_set_num)


def output_representation(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                          topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                          tau, model, diagnosis_size, read_from_cache, test_set_num):
    for dataset_name in 'hzsph', 'mimic-iii':
        logger.info('output representation: {}'.format(dataset_name))
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, False, False, True, test_set_num,
                     False, None, None)


def perplexity_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                        topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                        tau, model, diagnosis_size, read_from_cache, test_set_num):
    for dataset_name in 'hzsph', 'mimic-iii':
        logger.info('perplexity analysis: {}, {}'.format(dataset_name, topic_number_ntm))
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device, tau,
                     model, dataset_name, diagnosis_size, read_from_cache, False, False, True, False, test_set_num,
                     False, None, None)


def convergence_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                         device, tau, model, diagnosis_size, read_from_cache, test_set_num):
    for dataset_name in 'hzsph', 'mimic-iii':
        logger.info('convergence analysis: {}, {}'.format(dataset_name, topic_number_ntm))
        contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0, 0, 1
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num,
                     False, None, None)

        contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0.05, 0, 0.95
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num,
                     False, None, None)

        contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0.1, 0, 0.9
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num,
                     False, None, None)

        contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0.15, 0, 0.85
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num,
                     False, None, None)

        contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0.2, 0, 0.8
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                     topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                     tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num,
                     False, None, None)


def hyperparameter_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                            device, tau, model, diagnosis_size, read_from_cache, test_set_num, cut_length):
    # for similarity_coefficient in 0.15, :
    #     for contrastive_coefficient in 0, 0.05, 0.1, 0.15:
    #         for topic_coefficient in 0, 0.05, 0.1, 0.15:
    #             for dataset_name in 'hzsph', 'mimic-iii':
    #                 for test_set_num in 0, 1, 2, 3, 4:
    #                     ntm_coefficient = 1 - similarity_coefficient - topic_coefficient - contrastive_coefficient
    #                     logger.info('hyperparameter analysis: {}'.format(dataset_name))
    #                     one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
    #                                  epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
    #                                  ntm_coefficient, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
    #                                  False, False, False, False, test_set_num, False, None, cut_length)
    info_list = [
        [0, 0, 0, 0],
        [0, 0, 0.1, 2],
        [0, 0, 0.15, 1],
        [0, 0.05, 0.05, 3],
        [0, 0.05, 0.1, 2],
        [0, 0.05, 0.15, 2],

        [0, 0.05, 0.15, 4],
        [0, 0.1, 0.15, 2],
        [0, 0.15, 0.05, 2],
        [0.05, 0, 0.05, 2],
        [0.05, 0.05, 0.05, 0],
        [0.05, 0.05, 0.15, 3],

        [0.05, 0.1, 0.05, 4],
        [0.05, 0.1, 0.1, 1],
        [0.05, 0.15, 0.15, 3],
        [0.1, 0, 0, 1],
        [0.1, 0.1, 0, 4],
        [0.1, 0.15, 0, 3],

        [0.15, 0, 0.05, 3],
        [0.15, 0.05, 0, 0],
        [0.15, 0.05, 0.1, 2],
        [0.15, 0.1, 0.05, 4],
        [0.15, 0.1, 0.1, 2],
        [0.15, 0.15, 0.05, 4],
        [0.15, 0.15, 0.15, 4],
    ]
    dataset_name = 'hzsph'

    for info in info_list:
        similarity_coefficient, topic_coefficient, contrastive_coefficient, test_set_num = info
        ntm_coefficient = 1 - similarity_coefficient - topic_coefficient - contrastive_coefficient
        logger.info('hyperparameter analysis: {}'.format(dataset_name))
        one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                     epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                     ntm_coefficient, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
                     False, False, False, False, test_set_num, False, None, cut_length)


if __name__ == '__main__':
    main()
