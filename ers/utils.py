import numpy as np
import pickle


def compute_squared_distances(x, y):
    x2 = np.expand_dims(np.sum(x ** 2, axis=1), -1)
    y2 = np.expand_dims(np.sum(y ** 2, axis=1), -1)
    dists = -2 * np.dot(x, y.T) + y2.T + x2
    return dists


def pickle_obj(obj, fp):
    # open a file, where you ant to store the data
    file = open(fp, 'wb')

    # dump information to that file
    pickle.dump(obj, file)

    # close the file
    file.close()


def unpickle_obj(fp):
    # open a file, where you ant to store the data
    file = open(fp, 'rb')

    # dump information to that file
    obj = pickle.load(file)

    # close the file
    file.close()

    return obj
