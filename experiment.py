from train import one_cv_train
from config import args, logger


def main():
    batch_size, hidden_size_ntm, model, tau, diagnosis_size, topic_number_ntm, learning_rate, vocab_size_ntm, \
        epoch_number, contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient, device, \
        dataset_name, repeat_time, read_from_cache, test_set_num, experiment_type = \
        args['batch_size'], args['hidden_size_ntm'], args['model'], args['tau'], args['diagnosis_size'], \
        args['topic_number_ntm'], args['learning_rate'], args['vocab_size_ntm'],  args['epoch_number'], \
        args['contrastive_coefficient'], args['similarity_coefficient'], args['topic_coefficient'], \
        args['ntm_coefficient'], args['device'], args['dataset_name'], args['repeat_time'], args['read_from_cache'],\
        args['test_set_num'], args['experiment_type']

    for key in args:
        logger.info('{}: {}'.format(key, args[key]))

    for dataset_name in 'hzsph', 'mimic-iii':
        if experiment_type == 'hyperparameter':
            logger.info('hyperparameter analysis: {}, {}'.format(dataset_name, topic_number_ntm))
            for _ in range(args['repeat_time']):
                hyperparameter_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                        epoch_number, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
                                        test_set_num)
        else:
            logger.info('output representation: {}'.format(dataset_name))
            output_representation(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                  epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                                  ntm_coefficient, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
                                  test_set_num)

            for topic_number_ntm in 3, 5, 8, 10, 15, 20, 30:
                logger.info('perplexity analysis: {}, {}'.format(dataset_name, topic_number_ntm))
                perplexity_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                    epoch_number, topic_coefficient, contrastive_coefficient, similarity_coefficient,
                                    ntm_coefficient, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
                                    test_set_num)

            logger.info('convergence analysis: {}, {}'.format(dataset_name, topic_number_ntm))
            convergence_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm,
                                 epoch_number, device, tau, model, dataset_name, diagnosis_size, read_from_cache,
                                 test_set_num)


def output_representation(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                          topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                          tau, model, dataset_name, diagnosis_size, read_from_cache, test_set_num):
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, False, False, True, test_set_num)


def perplexity_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                        topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                        tau, model, dataset_name, diagnosis_size, read_from_cache, test_set_num):
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, False, True, False, test_set_num)


def convergence_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                         device, tau, model, dataset_name, diagnosis_size, read_from_cache, test_set_num):

    contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0, 0, 1
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num)

    contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0.05, 0, 0, 0.95
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num)

    contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0.05, 0, 0.95
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num)

    contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0, 0, 0.05, 0.95
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num)

    contrastive_coefficient, similarity_coefficient, topic_coefficient, ntm_coefficient = 0.05, 0.05, 0.05, 0.85
    one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                 topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient, device,
                 tau, model, dataset_name, diagnosis_size, read_from_cache, False, True, False, False, test_set_num)


def hyperparameter_analysis(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                            device, tau, model, dataset_name, diagnosis_size, read_from_cache, test_set_num):
    for similarity_coefficient in 0, 0.05, 0.1, 0.2:
        for topic_coefficient in 0, 0.05, 0.1, 0.2:
            for contrastive_coefficient in 0, 0.05, 0.1, 0.2:
                ntm_coefficient = 1 - similarity_coefficient - topic_coefficient - contrastive_coefficient
                one_cv_train(batch_size, hidden_size_ntm, topic_number_ntm, learning_rate, vocab_size_ntm, epoch_number,
                             topic_coefficient, contrastive_coefficient, similarity_coefficient, ntm_coefficient,
                             device, tau, model, dataset_name, diagnosis_size, read_from_cache, False, False, False,
                             False, test_set_num)


if __name__ == '__main__':
    main()
