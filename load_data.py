
import keras
from keras.datasets import mnist
import sys
from list_to_image import write_image


def main(argv):
    (images_train, label_train), (images_test, label_test) = mnist.load_data()


    write_image("testimage", images_train[1100])


    # images, labels = mndata.load_testing()


if __name__ == "__main__":
    main(sys.argv)