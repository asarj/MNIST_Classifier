from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cv2
import pickle
from tqdm import tqdm
import os
import warnings

class ImageDataset():
    """
    Class for holding image datasets that are loaded in manually
    Contains all batch preprocessing features as well
    """

    train = None
    x_train = None
    y_train = None

    test = None
    x_test = None
    y_test = None

    validation = None
    x_valid = None
    y_valid = None

    shape = None
    num_classes = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128


    def __init__(self, dir=None, type=None):
        """
        Constructor for building the TensorFlow dataset

        Loads the train, test, and validation pickle files from the specified directory and applies
        normalization on the dataset, followed by a setup of the tensorflow batch iterator

        :param str dir: the path to the dataset containing pickle files
        :param str type: the default tensorflow dataset to load in (only supports MNIST)
        """
        if type is not None:
            if type == 'MNIST':
                mnist = input_data.read_data_sets('data/MNIST/', one_hot=True)
                self.x_train = mnist.train.images
                self.y_train = mnist.train.labels
                self.x_train = self.x_train.astype(np.float32)
                self.shape = list(self.x_train.shape)
                print("Shape: ", self.shape)
                self.num_classes = self.y_train.shape[1]
                print("Unique Classes: ", self.num_classes)

                self.x_valid = mnist.validation.images
                self.y_valid = mnist.validation.labels
                self.x_valid = self.x_valid.astype(np.float32)

                self.x_test = mnist.test.images
                self.y_test = mnist.test.labels
                self.x_test = self.x_test.astype(np.float32)

        elif dir is not None:
            self.load(dir)

        print("Preprocessing train data...")
        self.normalize_image_pixels(self.x_train)
        print("Preprocessing test data...")
        self.normalize_image_pixels(self.x_test)
        print("Preprocessing validation data...")
        self.normalize_image_pixels(self.x_valid)
        self.setup_batch_iterator(self.x_train, self.y_train)

    def load(self, directory:str) -> None:
        """
        Populates the train, test, and validation global variables with raw data from the pickled files

        :param str directory: the path to the dataset containing pickle files
        :return: None, the class variables are populated accordingly
        """

        self.train = pickle.load(open(directory + 'train.p', 'rb'))
        self.x_train, self.y_train = self.train['features'], self.train['labels']
        self.x_train = self.x_train.astype(np.float32)

        self.shape = self.x_train[0].shape
        print("Shape: ", self.shape)
        self.num_classes = len(np.unique(self.y_train))
        print("Unique Classes: ", self.num_classes)

        self.test = pickle.load(open(directory + 'test.p', 'rb'))
        self.x_test, self.y_test = self.test['features'], self.test['labels']
        self.x_test = self.x_test.astype(np.float32)

        self.validation = pickle.load(open(directory + 'valid.p', 'rb'))
        self.x_valid, self.y_valid = self.validation['features'], self.validation['labels']
        self.x_valid = self.x_valid.astype(np.float32)

    def setup_batch_iterator(self, features:np.ndarray, labels:np.ndarray) -> None:
        """
        Constructs a TensorFlow dataset from the and sets up the batch iterator

        :param np.ndarray features: the images of the dataset
        :param np.ndarray labels: the labels for each feature
        :return: None, the batch iterator is constructed from tensors and stored in class variables
        """

        print("Setting up batch iterator...")

        data = tf.data.Dataset.from_tensor_slices((features, labels))
        data = data.shuffle(len(self.y_train), reshuffle_each_iteration=True).batch(self.batch_size)

        iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)
        self.train_init = iterator.make_initializer(data)
        self.x_batch, self.y_batch = iterator.get_next()

    def display_one(self, a:np.ndarray, title1="Original") -> None:
        """
        Helper function for displaying an image

        :param np.ndarray a: the image to display
        :param str title1: the title of the image to display when plotting
        :return: None, the image is rendered and shown
        """

        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def display_two(self, a:np.ndarray, b:np.ndarray, title1="Original", title2="Edited") -> None:
        """
        Helper function for displaying two images, usually for comparing before and after transformations

        :param np.ndarray a: the "before" image to display (should be the original image before any preprocessing
        :param np.ndarray b: the "after" image to display (after applying any transformations)
        :param str title1: the title of the "before" image to display when plotting
        :param str title2: the title of the "after" image to display when plotting
        :return: None, both images are rendered and shown
        """

        plt.subplot(121)
        plt.imshow(a)
        plt.title(title1)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(122)
        plt.imshow(b)
        plt.title(title2)
        plt.xticks([])
        plt.yticks([])
        plt.show()

    def preprocess(self, features:np.ndarray) -> np.ndarray:
        """
        Main function for preprocessing images

        :param np.ndarray features: the batch of images to perform preprocessing on
        :return: np.ndarray the modified batch of features
        """

        for i, img in (enumerate(features)):
            img = self.preprocess_improved(img)
            features[i] = img
        return features

    def preprocess_improved(self, image:np.ndarray) -> np.ndarray:
        """
        Main function for preprocessing images

        :param np.ndarray image: the image to apply a random transformation on
        :return: np.ndarray image the modified image
        """

        choice = randint(0, 3)
        if choice == 0:
            image = image
        elif choice == 1:
            image = self.perform_hist_eq(image)
        elif choice == 2:
            image = self.translate(image)
        elif choice == 3:
            image = self.gaussian(image)

        return self.normalize_image_pixels(image)

    def preprocess_normalize_only(self, features:np.ndarray) -> np.ndarray:
        """
        Main function for normalizing images only
        :param np.ndarray features: the batch of images to perform preprocessing on
        :return: np.ndarray the modified batch of features
        """

        for i, img in (enumerate(features)):
            img = self.normalize_image_pixels(img)
            features[i] = img

        return features

    def perform_hist_eq(self, image: np.ndarray) -> np.ndarray:
        """
        Takes in an image and performs histogram equalization -> improves contrast

        :param np.ndarray image: the image to apply histogram equalization on
        :return: np.ndarray image the modified image
        """

        R, G, B = cv2.split(image.astype(np.uint8))

        img_r = cv2.equalizeHist(R)
        img_g = cv2.equalizeHist(G)
        img_b = cv2.equalizeHist(B)

        image = cv2.merge((img_r, img_g, img_b))

        return image.astype(np.float32)

    def translate(self, image, height=32, width=32, max_trans=5) -> np.ndarray:
        """
        Applies a random translation in height and/or width

        :param np.ndarray image: the image to apply random translation on
        :param int height: the height of the image
        :param int width: the width of the image
        :param int max_trans: a max value to shift the height and width by
        :return: np.ndarray image the modified image
        """

        translate_x = max_trans * np.random.uniform() - max_trans / 2
        translate_y = max_trans * np.random.uniform() - max_trans / 2
        translation_mat = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        trans = cv2.warpAffine(image, translation_mat, (height, width))
        return trans

    def gaussian(self, image, ksize=(11, 11), border=0) -> np.ndarray:
        """
        Applies Gaussian Blur to the image

        :param np.ndarray image: the image to apply Gaussian Blur on
        :param tuple ksize: a tuple of 2 ints that describes the kernel size
        :param int border: value to apply a border on the image
        :return: np.ndarray image the modified image
        """

        return cv2.GaussianBlur(image, ksize, border)

    def normalize_image_pixels(self, image:np.ndarray) -> np.ndarray:
        """
        Function to normalize the image pixels. Assumes that the np.ndarray passed in contains values
        from [0,255] and normalizes it down to a value that is [0, 1)

        Revised to preprocess based on zero mean/unit variance, old code commented out

        :param np.ndarray image: the image to apply normalization on
        :return: np.ndarray image the modified image
        """

        # for normalizing pixels
        # return np.divide(image, 255.0)

        # for converting images to zero mean and unit variance
        # formula: z-score = x - mean / std
        # return (image - image.mean()) / image.std()
        return np.divide(np.subtract(image, np.mean(image)), np.std(image))


if __name__ == "__main__":
    """
    Main method for testing the data loader
    """

    # Ignore warnings
    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Testing GTSRB
    print("Testing GTSRB...")
    path = "GTSRB/"
    gtsrb = ImageDataset(path)
    n_train = len(gtsrb.x_train)
    n_valid = len(gtsrb.x_valid)
    n_test = len(gtsrb.x_test)
    image_shape = gtsrb.shape
    n_classes = gtsrb.num_classes

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    print()

    # Testing MNIST
    print("Testing MNIST...")
    mnist = ImageDataset(type="MNIST")
    n_train = len(mnist.x_train)
    n_valid = len(mnist.x_valid)
    n_test = len(mnist.x_test)
    image_shape = mnist.shape
    n_classes = mnist.num_classes

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)