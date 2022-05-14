import numpy as np
from skimage import io
from matplotlib import pyplot as plt
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
                image[i, j] = 0.1
            j += 1
        i += 1
    return image


def load_img_bw(path: str):
    image = io.imread(path, True)
    image = make_bw_image(image)
    return image


def get_ideal(y: int, n_classes):
    ideal = np.zeros((1, n_classes))
    ideal[0, y] = 1
    return ideal


def create_weights_matrix(num_of_cons, num_of_n):
    """Creates a new weight-matrix initialized with random values"""
    matrix = np.random.rand(num_of_cons, num_of_n)
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


def relu_d(x):
    return (x >= 0).astype(float)


def softmax(output):
    out = np.exp(output)
    return out / np.sum(out)


def cross_entropy(prediction, answer: int):
    return -np.log(prediction[0, answer])

# def cross_entropy(pred, ideal):
#     logvec = ideal * np.log(pred)
#     return -np.sum(logvec)


def plot_loss(n_epochs, loss):
    x = list(range(n_epochs*4))
    plt.plot(x, loss)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Loss (Calculated per sample)")
    plt.grid()
    plt.show()
