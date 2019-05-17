import os
import numpy as np
import tqdm as tqdm

from keras.preprocessing.image import load_img


def load_noisy_data(dir):
    files = os.listdir(dir)

    img = load_img(os.path.join(dir, files[0]))
    width, height = img.size
    data = np.zeros((len(files), width*height))

    labels = np.zeros(len(files), dtype=np.uint8)

    for id, file in enumerate(tqdm.tqdm(files)):
        labels[id] = str(file).split(".")[0].split("_")[1]
        img = load_img(os.path.join(dir, file), color_mode="grayscale")
        img = np.array(img)
        img = np.reshape(img, width*height)
        data[id] = img

    return data, labels
