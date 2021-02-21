import numpy as np
import sklearn

from sklearn.datasets import load_svmlight_file


def get_dataset(dataset, data_path='../datasets/'):
    if dataset in ['covtype', 'news20', 'real-sim', 'webspam', 'YearPredictionMSD']:
        return load_svmlight_file(data_path + dataset + '.bz2')
    elif dataset in ['a1a', 'a5a', 'a9a', 'mushrooms', 'gisette', 'w8a']:
        return load_svmlight_file(data_path + dataset)
    elif dataset == 'rcv1':
        return sklearn.datasets.fetch_rcv1(return_X_y=True, shuffle=False)
    elif dataset == 'YearPredictionMSD_binary':
        A, b = load_svmlight_file(data_path + dataset[:-7] + '.bz2')
        b = b > 2000
    elif dataset == 'news20_more_features':
        A, b = sklearn.datasets.fetch_20newsgroups_vectorized(return_X_y=True)
    elif dataset == 'news20_class1':
        A, b = load_svmlight_file(data_path + 'news20' + '.bz2')
        b = (b == 1)
    elif dataset == 'rcv1_binary':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
    return A, b
