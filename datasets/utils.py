import numpy as np
import sklearn


def get_dataset(dataset):
    if dataset == 'rcv1':
        return sklearn.datasets.fetch_rcv1(return_X_y=True)
    elif dataset == 'rcv1.binary':
        A, b = sklearn.datasets.fetch_rcv1(return_X_y=True)
        freq = np.asarray(b.sum(axis=0)).squeeze()
        main_class = np.argmax(freq)
        b = (b[:, main_class] == 1) * 1.
        b = b.toarray().squeeze()
        return A, b
