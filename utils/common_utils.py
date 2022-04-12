import numpy as np
import pickle


def save_sparse_csr(filename, array):
    array = array.todense()
    with open(filename, 'wb') as f:
        pickle.dump(array, f)


def load_sparse_csr(filename):
    with open(filename, 'rb') as f:
        x = pickle.load(f)
    return x
