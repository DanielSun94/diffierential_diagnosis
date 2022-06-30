import matplotlib.pyplot as plt
import pickle
import os
from util import dataset_selection
from config import representation_pkl, args, hzsph_tsne_cache, mimic_tsne_cache, pca_cache, logger
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import TSNE

hzsph_name_list = ['bipolar', 'depression', 'anxiety']
hzsph_colors = ["navy", "turquoise", "darkorange"]
mimic_colors = ["#1F77B4", '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22',
                '#17BECF']


def read_data(dataset, with_train, test_index=0):
    vocab_size, diagnosis_size, read_from_cache = \
        args['vocab_size_ntm'], args['diagnosis_size'], args['read_from_cache']
    representation_list = []
    label_list = []
    if dataset == 'hzsph':
        representation = pickle.load(open(representation_pkl.format('hzsph', '29062022124645'), 'rb'))
        five_fold_data = dataset_selection('hzsph', vocab_size, diagnosis_size, read_from_cache)[0]
        # representation_list = []
        # label_list = []
        # for i in range(5):
        #     if not (with_train or i == test_index):
        #         continue
        #     for sample in five_fold_data[i]:
        #         label_list.append(sample[3])
        # for sample in representation[1]:
        #     representation_list.append(sample)
        # if with_train:
        #     for sample in representation[0]:
        #         representation_list.append(sample)

    elif dataset == 'mimic':
        representation = pickle.load(open(representation_pkl.format('mimic-iii', '29062022164716'), 'rb'))
        five_fold_data = dataset_selection('mimic-iii', vocab_size, diagnosis_size, read_from_cache)[0]
        # representation_list = []
        # label_list = []
        # for i in range(5):
        #     if not (with_train or i == test_index):
        #         continue
        #     for sample in five_fold_data[i]:
        #         label_list.append(sample[3])
        #         representation_list.append(sample[2])
    else:
        raise ValueError('')

    representation_list = []
    label_list = []
    for i in range(5):
        if not (with_train or i == test_index):
            continue
        for sample in five_fold_data[i]:
            label_list.append(sample[3])
    for sample in representation[1]:
        representation_list.append(sample)
    if with_train:
        for sample in representation[0]:
            representation_list.append(sample)

    return representation_list, label_list


def fit_data(data, method='tsne'):
    representation_list, label_list = data[0], np.array(data[1])
    if method == 'pca':
        method = PCA(n_components=2)
        x_r = method.fit(representation_list).transform(representation_list)
    elif method == 'tsne':
        method = TSNE(n_components=2, learning_rate='auto', init='random')
        x_r = method.fit_transform(np.array(representation_list))
    else:
        raise ValueError('')
    return x_r, label_list


def plot(hzsph_tsne_data, hzsph_pca_data, mimic_tsne_data, mimic_pca_data):
    f, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1]})
    f.set_size_inches(3.4, 1.6)
    f.set_dpi(600)
    plt.subplots_adjust(bottom=0.01, right=0.99, top=0.99, left=0.01, wspace=0.04, hspace=0.01)
    plot_hzsph_tsne(hzsph_tsne_data, axs[0])
    plot_hzsph_pca(hzsph_pca_data, axs[1])
    # plot_mimic_tsne(mimic_tsne_data, axs[2])
    # plot_mimic_pca(mimic_pca_data, axs[3])
    plt.show()
    f.savefig("representation_visualization.svg")


def plot_hzsph_tsne(hzsph_tsne_data, subplot):
    representation, label_list = hzsph_tsne_data

    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines['bottom'].set_color('gray')
    subplot.spines['top'].set_color('gray')
    subplot.spines['right'].set_color('gray')
    subplot.spines['left'].set_color('gray')

    lw = 2
    for color, i in zip(hzsph_colors, [0, 1, 2]):
        subplot.scatter(
            representation[label_list == i, 0],
            representation[label_list == i, 1],
            s=0.05,
            color=color,
            alpha=0.8,
            lw=lw
        )


def plot_hzsph_pca(hzsph_pca_data, subplot):
    representation, label_list = hzsph_pca_data

    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_xlim(-0.15, 0.2)
    subplot.set_ylim(-0.15, 0.2)

    subplot.spines['bottom'].set_color('gray')
    subplot.spines['top'].set_color('gray')
    subplot.spines['right'].set_color('gray')
    subplot.spines['left'].set_color('gray')

    lw = 2
    for color, i in zip(hzsph_colors, [0, 1, 2]):
        subplot.scatter(
            representation[label_list == i, 0],
            representation[label_list == i, 1],
            s=0.05,
            color=color,
            alpha=0.8,
            lw=lw
        )


def plot_mimic_tsne(mimic_tsne_data, subplot):
    representation, label_list = mimic_tsne_data

    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.spines['bottom'].set_color('gray')
    subplot.spines['top'].set_color('gray')
    subplot.spines['right'].set_color('gray')
    subplot.spines['left'].set_color('gray')

    for color, i in zip(mimic_colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        subplot.scatter(
            representation[label_list == i, 0],
            representation[label_list == i, 1],
            s=0.05,
            color=color
        )


def plot_mimic_pca(mimic_pca_data, subplot):
    representation, label_list = mimic_pca_data

    subplot.set_xticks([])
    subplot.set_yticks([])
    # subplot.set_xlim(-0.15, 0.2)
    # subplot.set_ylim(-0.15, 0.2)

    subplot.spines['bottom'].set_color('gray')
    subplot.spines['top'].set_color('gray')
    subplot.spines['right'].set_color('gray')
    subplot.spines['left'].set_color('gray')

    for color, i in zip(mimic_colors, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
        subplot.scatter(
            representation[label_list == i, 0],
            representation[label_list == i, 1],
            s=0.05,
            color=color,
        )


def read_representation(hzsph_tsne_cache_flag, mimic_tsne_cache_flag):
    if os.path.exists(pca_cache):
        mimic_pca_data, hzsph_pca_data = pickle.load(open(pca_cache, 'rb'))
        logger.info('load pca success')
    else:
        hzsph_data = read_data('hzsph', True, test_index=0)
        logger.info('read hzsph success')
        mimic_data = read_data('mimic', False, test_index=0)
        logger.info('read mimic success')
        hzsph_pca_data = fit_data(hzsph_data, method='pca')
        logger.info('fit hzsph pca success')
        mimic_pca_data = fit_data(mimic_data, method='pca')
        logger.info('fit mimic pca success')
        pickle.dump((mimic_pca_data, hzsph_pca_data), open(pca_cache, 'wb'))
        logger.info('dump pca success')
    if hzsph_tsne_cache_flag and os.path.exists(hzsph_tsne_cache):
        hzsph_tsne_data = pickle.load(open(hzsph_tsne_cache, 'rb'))
        logger.info('load hzsph tsne success')
    else:
        hzsph_data = read_data('hzsph', True, test_index=0)
        logger.info('read hzsph success')
        hzsph_tsne_data = fit_data(hzsph_data, method='tsne')
        logger.info('fit hzsph tsne success')
        pickle.dump(hzsph_tsne_data, open(hzsph_tsne_cache, 'wb'))
        logger.info('dump hzsph tsne success')
    if mimic_tsne_cache_flag and os.path.exists(mimic_tsne_cache):
        mimic_tsne_data = pickle.load(open(mimic_tsne_cache, 'rb'))
        logger.info('load mimic tsne success')
    else:
        mimic_data = read_data('mimic', False, test_index=0)
        logger.info('read mimic success')
        mimic_tsne_data = fit_data(mimic_data, method='tsne')
        logger.info('fit mimic tsne success')
        pickle.dump(mimic_tsne_data, open(mimic_tsne_cache, 'wb'))
        logger.info('dump mimic tsne success')
    return mimic_pca_data, mimic_tsne_data, hzsph_pca_data, hzsph_tsne_data


def main():
    hzsph_tsne_cache_flag, mimic_tsne_cache_flag = True, False
    mimic_pca_data, mimic_tsne_data, hzsph_pca_data, hzsph_tsne_data = \
        read_representation(hzsph_tsne_cache_flag, mimic_tsne_cache_flag)
    logger.info('data loaded')
    plot(hzsph_tsne_data, hzsph_pca_data, mimic_tsne_data, mimic_pca_data)
    logger.info('exit')


if __name__ == '__main__':
    main()
