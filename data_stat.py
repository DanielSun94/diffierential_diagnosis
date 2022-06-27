import numpy as np
import pickle
from config import mimic_iii_cache_3, hzsph_cache


def main():
    mimic_iii = pickle.load(open(mimic_iii_cache_3, 'rb'))
    hzsph = pickle.load(open(hzsph_cache, 'rb'))

    mimic_iii_data = []
    for fold in mimic_iii[4]:
        for sample in fold:
            mimic_iii_data.append(sample[1])
    hzsph_data = []
    for fold in hzsph[0]:
        for sample in fold:
            hzsph_data.append(sample[1])
    print('')
    print('hzsph_data')
    stat(hzsph_data)
    print('mimic_iii_data')
    stat(mimic_iii_data)


def stat(data):
    print('sample number: {}'.format(len(data)))

    token_number = []
    for sample in data:
        token_number.append(np.sum(sample))

    print('average tokens: {}'.format(np.average(token_number)))
    print('median tokens: {}'.format(np.median(token_number)))
    print('max tokens: {}'.format(np.max(token_number)))
    print('min tokens: {}'.format(np.min(token_number)))
    print('25% tokens: {}'.format(np.quantile(token_number, 0.25)))
    print('75% tokens: {}'.format(np.quantile(token_number, 0.75)))


if __name__ == '__main__':
    main()
