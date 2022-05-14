import sys
import random
import numpy as np
from matplotlib import pyplot as plt
import help_funcs as myfunc
# np.set_printoptions(threshold=sys.maxsize)

# Initialize the input
images = []
image = myfunc.load_img_bw("./imgsrc/s1.png")
images.append(image)
image = myfunc.load_img_bw("./imgsrc/s2.png")
images.append(image)
image = myfunc.load_img_bw("./imgsrc/s3.png")
images.append(image)
image = myfunc.load_img_bw("./imgsrc/s4.png")
images.append(image)

classes = ["Bobsled", "Slide", "Skates", "Fight"]
class_id = {"Bobsled": 0, "Slide": 1, "Skates": 2, "Fight": 3}

dataset = list(zip(class_id, images))

# Hyperparameters
INPUT_VEC_SZ = image[:, 0].size  # Square image gets sliced and
L1_COUNT = image[:, 0].size      # being fed to the net, one to each neuron
L2_COUNT = 16                    # May vary
L3_COUNT = 4                     # Number of classes

LRATE = 0.1     # Learning rate
EPOCHS = 100    # Number of epochs

# Initialize the network with random weights
m_w1 = myfunc.create_weights_matrix(INPUT_VEC_SZ, L1_COUNT)
m_w2 = myfunc.create_weights_matrix(L1_COUNT, L2_COUNT)
m_w3 = myfunc.create_weights_matrix(L2_COUNT, L3_COUNT)
# Biases for layer 2 and 3
l2_bias = np.random.rand(L2_COUNT)
l3_bias = np.random.rand(L3_COUNT)

# Layer 1 sum(linear)- and output(non-linear)- vectors
l1_output = np.ndarray((1, L1_COUNT))
l1_sum = np.ndarray((1, L1_COUNT))
# Layer 2 sum(linear)- and output(non-linear)- vectors
l2_output = np.ndarray((1, L2_COUNT))
l2_sum = np.ndarray((1, L2_COUNT))
# Layer 3 sum(linear)- and output(non-linear)- vectors
l3_output = np.ndarray((1, L3_COUNT))
l3_sum = np.ndarray((1, L3_COUNT))

loss = []
for epoch in range(EPOCHS):
    print(f"Epoch: <{epoch}>")
    random.shuffle(dataset)
    s_cnt: int = 0  # Sample counter
    for s_class, s_image in dataset:
        for cnt in range(L1_COUNT):
            l1_sum[0, cnt] = np.sum(s_image[:, cnt] * m_w1[:, cnt])
        l1_output = myfunc.relu_activation_vec(l1_sum)

        l2_sum = l1_output @ m_w2 + l2_bias
        l2_output = myfunc.relu_activation_vec(l2_sum)

        l3_sum = l2_output @ m_w3 + l3_bias
        l3_output = myfunc.softmax(l3_sum)

        error = myfunc.cross_entropy(l3_output, class_id[s_class])
        loss.append(error)

        # BACKWARD
        d_out = l3_output - myfunc.get_ideal(class_id[s_class], len(classes))
        d_w3 = l2_output.T @ d_out
        d_b3 = d_out

        d_l2 = (d_out @ m_w3.T) * myfunc.relu_d(l2_sum)
        d_w2 = l1_output.T @ d_l2
        d_b2 = d_l2

        # is delta from previous vids?
        d_w1 = np.ndarray((54, 54))
        d_l1 = (d_l2 @ m_w2.T) * myfunc.relu_d(l1_sum)
        for cnt in range(L1_COUNT):
            d_w1[cnt] = np.reshape(s_image[:, cnt], (1, 54)) * d_l1

        m_w3 = m_w3 - LRATE * d_w3
        l3_bias = l3_bias - LRATE * d_b3

        m_w2 = m_w2 - LRATE * d_w2
        l2_bias = l2_bias - LRATE * d_b2

        m_w1 = m_w1 - LRATE * d_w1

        s_cnt += 1


def predict(x):
    for cnt in range(L1_COUNT):
        l1_sum[0, cnt] = np.sum(x[:, cnt] * m_w1[:, cnt])
    l1_output = myfunc.relu_activation_vec(l1_sum)

    l2_sum = l1_output @ m_w2 + l2_bias
    l2_output = myfunc.relu_activation_vec(l2_sum)

    l3_sum = l2_output @ m_w3 + l3_bias
    l3_output = myfunc.softmax(l3_sum)
    return l3_output


def acc():
    correct = 0
    for s_c, s_img in dataset:
        output = predict(s_img)
        pred = np.argmax(output)
        if pred == class_id[s_c]:
            correct += 1
    return correct / len(dataset)


print(acc())
myfunc.plot_loss(EPOCHS, loss)
