import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import help_funcs as myfunc

# Initialize the input
image: np.ndarray = io.imread("./imgsrc/s1.png", True)
image = myfunc.make_bw_image(image)
trainset = [image]

image: np.ndarray = io.imread("./imgsrc/s2.png", True)
image = myfunc.make_bw_image(image)
trainset.append(image)

image: np.ndarray = io.imread("./imgsrc/s3.png", True)
image = myfunc.make_bw_image(image)
trainset.append(image)

image: np.ndarray = io.imread("./imgsrc/s4.png", True)
image = myfunc.make_bw_image(image)
trainset.append(image)

INPUT_VEC_SZ = image[:, 0].size  # Square image gets sliced and
L1_COUNT = image[:, 0].size      # being fed to the net one to each neuron
L2_COUNT = 8                     # May vary
L3_COUNT = 4                     # Number of classes

# Initialize the network with random weights
m_w1 = myfunc.create_weights_matrix(INPUT_VEC_SZ, L1_COUNT)
m_w2 = myfunc.create_weights_matrix(L1_COUNT, L2_COUNT)
m_w3 = myfunc.create_weights_matrix(L2_COUNT, L3_COUNT)

# Biases for layer 2 and 3
l2_bias = np.random.rand(L2_COUNT)
l3_bias = np.random.rand(L3_COUNT)

# Ideal values vector and class names
class_names = ["Bobsled", "Slide", "Skates", "Fight"]
m_ideal = np.array([
    [0.9, 0.1, 0.1, 0.1],
    [0.1, 0.9, 0.1, 0.1],
    [0.1, 0.1, 0.9, 0.1],
    [0.1, 0.1, 0.1, 0.9]])

l1_output = np.ndarray(L1_COUNT)
for t_sample in trainset:
    for cnt in range(L1_COUNT):
        # ZERO SHOULD BE CHANGED
        l1_output[cnt] = t_sample[:, cnt] @ m_w1[:, cnt]
    l1_output = myfunc.relu_activation_vec(l1_output)

    l2_output = l1_output @ m_w2 + l2_bias
    l2_output = myfunc.relu_activation_vec(l2_output)

    l3_output = l2_output @ m_w3 + l3_bias
    l3_output = myfunc.softmax(l3_output)
    # ZERO SHOULD BE CHANGED
    error = myfunc.cross_entropy(l3_output, m_ideal[trainset.index(t_sample)])
