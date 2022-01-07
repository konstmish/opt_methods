import numpy as np
import sklearn

from sklearn.datasets import load_svmlight_file


def get_dataset(dataset, data_path='../optmethods/datasets/'):
    if len(data_path) > 0 and data_path[-1] != '/':
        data_path += '/'
    if dataset in ['news20', 'real-sim', 'webspam', 'YearPredictionMSD']:
        return load_svmlight_file(data_path + dataset + '.bz2')
    elif dataset in ['a1a', 'a5a', 'a9a', 'mushrooms', 'gisette', 'w8a']:
        return load_svmlight_file(data_path + dataset)
    elif dataset == 'covtype':
        return sklearn.datasets.fetch_covtype(return_X_y=True)
    elif dataset == 'covtype_binary':
        A, b = sklearn.datasets.fetch_covtype(return_X_y=True)
        # Following paper "A Parallel Mixture of SVMs for Very Large Scale Problems"
        # we make the problem binary by splittong the data into class 2 and the rest.
        b = (b == 2)
    elif dataset == 'YearPredictionMSD_binary':
        A, b = load_svmlight_file(data_path + dataset[:-7] + '.bz2')
        b = b > 2000
    elif dataset == 'news20_more_features':
        A, b = sklearn.datasets.fetch_20newsgroups_vectorized(return_X_y=True)
    elif dataset == 'news20_binary':
        A, b = load_svmlight_file(data_path + 'news20' + '.bz2')
        b = (b == 1)
    elif dataset == 'rcv1':
        return sklearn.datasets.fetch_rcv1(return_X_y=True)
    elif dataset == 'rcv1_binary':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
    else:
        raise ValueError(f'The dataset {dataset} is not supported. Consider loading it yourself.')
    return A, b
