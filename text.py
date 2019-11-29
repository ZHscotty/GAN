from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.datasets import mnist
"""
im = Image.fromarray(np.uint8(inputs))
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[0]
# print(img)
# img = np.random.uniform(-1, 1, size=(100, 100))
# print(img.shape)
img_new = img/255
im = Image.fromarray(np.uint8(img_new))
im.show()
# plt.show(img)