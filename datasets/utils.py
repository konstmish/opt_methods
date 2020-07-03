import numpy as np
import sklearn
from sklearn.datasets import load_svmlight_file


def get_dataset(dataset, data_path='../datasets/'):
    if dataset in ['covtype', 'real-sim', 'webspam', 'YearPredictionMSD']:
        return load_svmlight_file(data_path + dataset + '.bz2')
    elif dataset in ['mushrooms', 'gisette', 'w8a']:
        return load_svmlight_file(data_path + dataset)
    elif dataset == 'rcv1':
        return sklearn.datasets.fetch_rcv1(return_X_y=True)
    elif dataset == 'YearPredictionMSD_binary':
        A, b = load_svmlight_file(data_path + dataset + '.bz2')
        b = b > 2000
        return A, b
    elif dataset == 'rcv1_binary':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
        return A, b
