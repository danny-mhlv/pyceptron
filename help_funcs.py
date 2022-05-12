from cmath import log
from distutils.log import Log
import numpy as np

E = 2.71828182


def make_bw_image(image: np.ndarray):
    """Makes a given image black and white"""

    i, j = 0, 0
    for row in image:
        j = 0
        for pixel_value in row:
            if pixel_value < 0.5:
                image[i, j] = 0.0
            else:
                image[i, j] = 1.0
            j += 1
        i += 1
    return image


def create_weights_matrix(num_of_neurons, num_of_connections):
    """Creates a new weight-matrix initialized with random values"""
    matrix = np.random.rand(num_of_neurons, num_of_connections)
    return matrix


def sigmoid_activation_vec(x: np.ndarray, coeff):
    output = np.ndarray(x.shape)
    cnt: int = 0
    for element in x:
        output[cnt] = (1 / (1 + (E ** (-coeff * element))))
        cnt += 1
    return output


def sigmoid_activation(x, coeff):
    return 1 / (1 + np.exp(-coeff * x))


def relu_activation_vec(x):
    return np.maximum(x, 0)


def softmax(output):
    out = np.exp(output)
    return out / np.sum(out)


def cross_entropy(pred, ideal):
    logvec = ideal * np.log(pred)
    return -np.sum(logvec)
