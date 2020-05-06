import numpy as np
import tensorflow as tf
import os
import warnings
from data_loader import ImageDataset
from datetime import datetime
import matplotlib.pyplot as plt
from mnist_cnn_classifier import CNN as MNISTLeNet
import tensorflow.contrib.slim as slim


class MixtureOfExperts():
    """
    Class for building and training a mixture of experts model on an image dataset
    Performs a basic model construction that goes as follows:
    logits from all experts -> gating/activations and distributions -> MoE logits -> softmax/cross entropy
    with Adam Optimizer

    Implemented in this classifier:
    - Early stopping
    - Learning rate estimator
    - L2 regularization
    """

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128
    learning_rate = 0.001
    num_experts = None
    num_inputs = 400
    networks = list()

    def __init__(self, dataset: ImageDataset, num_experts=4, load_dataset=True, num_epochs=100, learning_rate=0.001,
                 enable_session=False, dynamic_lr=True, shape=None, num_classes=None):
        """
        Constructor for building the CNN classifier
        :param ImageDataset dataset: a fully constructed ImageDataset object with batch iterator set up
        :param int num_experts: the number of experts to create the model with
        :param bool load_dataset: boolean flag to determine whether to load the dataset object passed in
                                  set to False if using this model with something like a mixture of experts or ensemble
        :param int num_epochs: number of epochs to use for BOTH finding optimal learning rate and training the model
        :param float learning_rate: default/starting learning rate to train the model on. This value will change during
                              model construction if the dynamic_lr flag is set to True
        :param bool enable_session: boolean flag to determine whether to start a session with this model
                                    set to False if using this model with something like a mixture of experts or
                                    ensemble
        :param bool dynamic_lr: boolean flag to determine whether to find the optimal learning rate before training
                                the model
        :param tuple shape: hard-coded shape of the object if the flag load_dataset is set to false
        :param int num_classes: hard-coded number of labels of the object if the flag load_dataset is set to false
        """

        if enable_session:
            self.tf_sess = tf.Session()

        if load_dataset:
            self.dataset = dataset

        # Make sure the number of experts passed in is greater than 2
        if num_experts < 2:
            raise Exception("Mixture of Experts model needs 2 or more experts!")

        self.num_inputs = self.dataset.x_train.shape[1]
        self.num_experts = num_experts
        self.learning_rate = learning_rate

        self.build_model(epochs=num_epochs,
                         learning_rate=learning_rate,
                         enable_dynamic_lr=dynamic_lr,
                         dataset_loaded=load_dataset,
                         shape=shape,
                         num_classes=num_classes)

    def build_model(self, epochs=50, learning_rate=0.001,
                    enable_dynamic_lr=True, dataset_loaded=True, shape=None, num_classes=None):
        """
        Builds the model layers and finds the optimal learning rate for training if specified
        :param int epochs: number of epochs to use for finding optimal learning rate
        :param float learning_rate: starting learning rate to train the model on. This value will change during
                              model construction if the dynamic_lr flag is set to True
        :param bool enable_dynamic_lr: boolean flag to determine whether to find the optimal learning rate before
                                       training the model
        :param bool dataset_loaded: boolean flag to determine whether to load the dataset object passed in. Set to
                                    False if using this model with something like a mixture of experts or ensemble
                                    (though this would have been set in the constructor anyway)
        :param tuple shape: hard-coded shape of the object if the flag dataset_loaded is set to false
        :param int num_classes: hard-coded number of labels of the object if the flag dataset_loaded is set to false
        :return: None, model is constructed accordingly
        """

        with tf.variable_scope('moe', reuse=tf.AUTO_REUSE) as scope:
            self.x = tf.placeholder(tf.float32, [None, self.num_inputs], name='x')
            print(self.x)
            self.y = tf.placeholder(tf.float32, [None, self.dataset.num_classes], name='y')
            print(self.y)

        if enable_dynamic_lr:
            # Get optimal learning rate for LeNet classifiers
            print("Getting optimal learning rate for LeNet classifiers...")
            print()
            lenet = MNISTLeNet(dataset=self.dataset,
                               load_dataset=True,
                               num_epochs=50,
                               learning_rate=learning_rate,
                               enable_session=enable_session,
                               dynamic_lr=enable_dynamic_lr,
                               shape=shape,
                               num_classes=num_classes)

            self.learning_rate = lenet.learning_rate
            print(f"Optimal Learning Rate for All LeNet models (total = {self.num_experts}): {self.learning_rate}")
            print()

            # Don't need these anymore
            del lenet

        for x in range(self.num_experts):
            self.networks.append(
                MNISTLeNet(dataset=self.dataset,
                           load_dataset=False,
                           num_epochs=epochs,
                           learning_rate=self.learning_rate,
                           enable_session=False,
                           dynamic_lr=False,
                           shape=self.dataset.shape,
                           num_classes=self.dataset.num_classes)
            )

        print("Building model...")
        concat = tf.concat([expert.logits for expert in self.networks], axis=1)
        gate_activations = self.fc_layer(input=None, inputs=concat,
                                         outputs=self.dataset.num_classes * (self.num_experts + 1),
                                         relu=False, is_linear=True)
        print("Gate Activation shape:", gate_activations.shape)

        gating_distribution = tf.nn.softmax(tf.reshape(gate_activations, [-1, self.num_experts + 1]))
        print("Gate Distribution shape:", gating_distribution.shape)

        expert_activations = self.fc_layer(input=None, inputs=concat,
                                           outputs=self.dataset.num_classes * self.num_experts,
                                           relu=False, is_linear=True)
        print("Expert Activation shape:", expert_activations.shape)

        expert_distribution = tf.nn.sigmoid(tf.reshape(expert_activations, [-1, self.num_experts]))
        print("Expert Distribution shape:", expert_distribution.shape)

        final_probabilities = tf.reduce_sum(gating_distribution[:, :self.num_experts] * expert_distribution, 1)
        print("Final probabilities shape: ", final_probabilities.shape)

        self.logits = tf.reshape(final_probabilities, [-1, self.dataset.num_classes])
        print("Logits shape:", self.logits.shape)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        self.loss = tf.reduce_mean(cross_entropy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.predict = tf.nn.softmax(self.logits)

    def fc_layer(self, input, inputs, outputs, relu=True, is_linear=False):
        """
        Creates a fully connected layer for the model. Doubles as a method to construct a linear layer
        if the is_linear flag is set to true

        :param input: the input to feed into the layer if creating a fully connected layer
        :param inputs: the numpy array to create the weights with and perform linear layer construction if
                       the is_linear flag is set to True
        :param outputs: the numpy array to create the weights and biases
        :param relu: boolean flag to determine whether to apply a relu to this model
        :param is_linear: boolean flag to determine whether to treat this layer as a linear layer
        :return: the constructed layer
        """

        layer = None

        if is_linear:
            weights = self.create_weights(shape=[inputs.get_shape()[-1], outputs])
            biases = self.create_biases(outputs)

            layer = tf.matmul(inputs, weights)

        else:
            weights = self.create_weights(shape=[inputs, outputs])
            biases = self.create_biases(outputs)

            layer = tf.matmul(input, weights)

        layer += biases

        if relu:
            layer = tf.nn.relu(layer)

        return layer

    def create_weights(self, shape: list, stddev=0.05) -> tf.Variable:
        """
        Constructs the weights in the form of a TensorFlow variable

        :param shape: the shape of the weights variable
        :param stddev: the standard deviation for the weights
        :return: tf.Variable the weights in a TensorFlow variable
        """

        return tf.Variable(tf.truncated_normal(shape=shape, mean=0, stddev=stddev))

    def create_biases(self, size: int) -> tf.Variable:
        """
        Constructs the weights in the form of a TensorFlow variable

        :param size: the number of zeros to hold the biases
        :return: tf.Variable the biases in a TensorFlow variable
        """

        return tf.Variable(tf.zeros([size]))

    def train_model(self, epochs: int, limit=6):
        """
        Trains the model and evaluates the train and valdiation accuracy with every epoch. If early stopping kicks
        in, the entire test set is evaluated accordingly

        :param int epochs: number of epochs to use for training the model
        :param int limit: the tolerance limit for determining whether to apply early stopping during training
        :return: None, model stats are shown accordingly
        """

        print("Training model...")
        self.tf_sess.run(tf.global_variables_initializer())

        best, no_change, total_loss, total_acc = 0, 0, 0, 0

        for epoch in range(epochs):
            self.tf_sess.run(self.dataset.train_init)
            try:
                total = 0
                while 1:
                    bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])

                    feed_dict = {
                        self.x: bx,
                        self.y: by
                    }
                    self.tf_sess.run(self.optimizer, feed_dict=feed_dict)
                    loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
                    total += acc * len(by)
                    total_loss += loss * len(by)
                    total_acc += acc * len(by)

            except(tf.errors.OutOfRangeError):
                pass

            feed_dict = {
                self.x: self.dataset.x_valid,
                self.y: self.dataset.y_valid
            }

            vloss, vacc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            print(f'epoch {epoch + 1}: loss = {vloss:.4f}, '
                  f'training accuracy = {total / len(self.dataset.y_train):.4f}, '
                  f'validation accuracy = {vacc:.4f}, '
                  f'learning rate = {self.learning_rate:.10f}')
            # Early stopping
            if vacc > best:
                best = vacc
                no_change = 0
            else:
                no_change += 1

            if no_change >= limit:
                print("Early stopping...")
                break

        feed_dict = {
            self.x: self.dataset.x_test,
            self.y: self.dataset.y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy = {acc:.4f}')

        # Evaluate performance of each expert
        for index, expert in enumerate(self.networks):
            loss, acc = self.tf_sess.run([expert.loss, expert.accuracy], feed_dict=feed_dict)
            print(f'\tExpert {index + 1}: test loss = {loss:.4f}, test accuracy = {acc:.4f}')


if __name__ == "__main__":
    """
    Main method for testing the Mixture of Experts model
    """

    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("TensorFlow Version: ", tf.__version__)

    # path = "GTSRB/"
    # gtsrb = ImageDataset(path)
    # n_train = len(gtsrb.x_train)
    # n_valid = len(gtsrb.x_valid)
    # n_test = len(gtsrb.x_test)
    # image_shape = gtsrb.shape
    # n_classes = gtsrb.num_classes
    #
    # epochs = 50
    # learning_rate = 1e-3
    # enable_session = True
    # num_experts = 4
    # dynamic_lr = True
    # shape = gtsrb.shape
    # num_classes = gtsrb.num_classes

    # Trying MNIST now
    mnist = ImageDataset(type='MNIST')
    n_train = len(mnist.x_train)
    n_valid = len(mnist.x_valid)
    n_test = len(mnist.x_test)
    image_shape = mnist.shape
    n_classes = mnist.num_classes

    epochs = 500
    learning_rate = 1e-3
    enable_session = True
    num_experts = 8
    dynamic_lr = False
    shape = mnist.shape
    num_classes = mnist.num_classes

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print()

    start = datetime.now()
    moe = MixtureOfExperts(dataset=mnist,
                           num_experts=num_experts,
                           num_epochs=epochs,
                           learning_rate=learning_rate,
                           enable_session=enable_session,
                           dynamic_lr=dynamic_lr,
                           shape=shape,
                           num_classes=num_classes)
    if enable_session:
        moe.train_model(epochs=epochs, limit=8)
        end = datetime.now()
        print("Time taken to build and train the model on " + str(epochs) + " epochs is:", str(end - start))
        moe.tf_sess.close()
    else:
        end = datetime.now()
        print("Time taken to build the model on " + str(epochs) + " epochs is:", str(end - start))