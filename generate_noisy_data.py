import os

import numpy as np
import tqdm as tqdm
from keras.datasets import mnist
from PIL import Image

HEIGHT = 28
WIDTH = 28

(images_train, label_train), (images_test, label_test) = mnist.load_data()


path = "/home/komputerka/PycharmProjects/mnist19/samples/noisy"
noisy_img_train = []
it = 0
for image, label in tqdm.tqdm(zip(images_train, label_train)):
    img = np.array(image)
    noise = np.random.randint(5, size=(28, 28), dtype='uint8')
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if img[i][j] != 255:
                img[i][j] += noise[i][j]
    im = Image.fromarray(img)
    im.save(os.path.join(path, str(it) + "_" + str(label) + ".jpg"))
    it += 1
    # plt.imshow(img, cmap='Greys')
    # plt.show()


