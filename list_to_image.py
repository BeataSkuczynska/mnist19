from PIL import Image
import numpy as np


def write_image(filename, image):
    im = np.array(image).reshape(-1, 28)

    I8 = (((im - im.min()) / (im.max() - im.min())) * 255.9).astype(np.uint8)

    img = Image.fromarray(I8)
    img.save("{0}.png".format(filename))