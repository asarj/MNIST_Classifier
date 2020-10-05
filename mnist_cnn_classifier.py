import numpy as np
import tensorflow as tf
import os
import warnings
from data_loader import ImageDataset
from datetime import datetime
import matplotlib.pyplot as plt


class CNN():
    """
    Class for building and training a CNN model on an image dataset (in this case, meant for MNIST)
    Performs a LeNet-based model construction that goes as follows:
    convolutional layer -> maxpool -> convolutional layer -> maxpool -> flattened layer -> fully connected layer
    -> fully connected layer -> logits -> softmax/cross entropy
    with Adam Optimizer

    Implemented in this classifier:
    - Early stopping
    - Learning rate estimator
    """

    tf_sess = None
    model = None
    dataset = None
    batch_size = 128
    repeat_size = 5
    shuffle = 128
    learning_rate = 0.001

    def __init__(self, dataset:ImageDataset, load_dataset=True, num_epochs=100, learning_rate=0.001,
                 enable_session=False, dynamic_lr=True, shape=None, num_classes=None):
        """
        Constructor for building the CNN classifier
        :param ImageDataset dataset: a fully constructed ImageDataset object with batch iterator set up
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

        print("Building model...")
        x, y, cross_entropy, correct = None, None, None, None
        if dataset_loaded:
            self.x = tf.placeholder(tf.float32, [None, self.dataset.shape[1]])
            self.y = tf.placeholder(tf.float32, [None, self.dataset.num_classes])
            print("Shape of initial layer:", self.x.shape)
        else:
            x = tf.get_default_graph().get_tensor_by_name('moe/x:0')
            y = tf.get_default_graph().get_tensor_by_name('moe/y:0')
            print("Shape of initial layer:", x.shape)


        # Reshaping
        if dataset_loaded:
            reshaped = tf.reshape(self.x, shape=[-1, 28, 28, 1])
        else:
            reshaped = tf.reshape(x, shape=[-1, 28, 28, 1])

        # First layer
        c1_channels = 1
        c1_filters = 6
        c1 = self.conv_layer(input=reshaped, input_channels=c1_channels, filters=c1_filters, filter_size=5)

        print("Shape of After 1st layer:", c1.shape)
        # Pooling
        pool1 = self.pool(layer=c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        print("Shape of After 1st pooling:", pool1.shape)

        # Second layer
        c2_channels = 6
        c2_filters = 16
        c2 = self.conv_layer(input=pool1, input_channels=c2_channels, filters=c2_filters, filter_size=5)
        print("Shape of After 2nd layer:", c2.shape)

        # Pooling
        pool2 = self.pool(layer=c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])
        print("Shape of After 2nd pooling:", pool2.shape)

        # Flattened layer
        flattened = self.flatten_layer(layer=pool2)
        print("Shape of After flattening:", flattened.shape)

        # First Fully Connected Layer
        fc1_input = 256
        fc1_output = 120
        fc1 = self.fc_layer(input=flattened, inputs=fc1_input, outputs=fc1_output, relu=True)
        print("Shape of After 1st FC:", fc1.shape)

        # Second Fully Connected Layer
        fc2_input = 120
        fc2_output = 84
        fc2 = self.fc_layer(input=fc1, inputs=fc2_input, outputs=fc2_output, relu=True)
        print("Shape of After 2nd FC:", fc2.shape)

        # Logits
        l_inp = 84
        if dataset_loaded:
            l_out = self.dataset.num_classes
        else:
            l_out = num_classes
        self.logits = self.fc_layer(input=fc2, inputs=l_inp, outputs=l_out, relu=False)
        print("Shape after logits:", self.logits.shape)
        print()

        if dataset_loaded:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y)
        else:
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y)

        self.loss = tf.reduce_mean(cross_entropy)

        if dataset_loaded:
            correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
        else:
            correct = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(y, axis=1))

        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        self.prediction = tf.argmax(self.logits, axis=1)

        # Get starting learning rate
        if enable_dynamic_lr:
            self.learning_rate = self.get_optimal_learning_rate(epochs=epochs,
                                                                learning_rate=learning_rate,
                                                                plot_charts=False)
        else:
            self.learning_rate = learning_rate

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def train_model(self, epochs:int, limit=6):
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
                self.x: self.dataset.preprocess_normalize_only(self.dataset.x_valid),
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
            self.x: self.dataset.preprocess_normalize_only(self.dataset.x_test),
            self.y: self.dataset.y_test
        }
        acc = self.tf_sess.run(self.accuracy, feed_dict=feed_dict)
        print(f'test accuracy = {acc:.4f}')

    def get_optimal_learning_rate(self, epochs=50, learning_rate=1e-5, plot_charts=False):
        """
        Finds the optimal learning rate of the model using gradient descent before using this value in the model's
        optimizer

        :param int epochs: number of epochs to use for finding optimal learning rate
        :param float learning_rate: starting learning rate to train the model on
        :param bool plot_charts: boolean flag to determine whether to display the resulting charts for finding the
                                 optimal learning rate
        :return: float the optimal learning rate
        """

        print("Finding optimal learning rate...")
        self.tf_sess.run(tf.global_variables_initializer())
        rates = list()
        t_loss = list()
        t_acc = list()

        self.tf_sess.run(self.dataset.train_init)
        for i in range(epochs):

            learning_rate *= 1.1
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
            bx, by = self.tf_sess.run([self.dataset.x_batch, self.dataset.y_batch])
            feed_dict = {
                self.x: bx,#np.reshape(bx, (-1, 28, 28, 1)),
                self.y: by
            }

            self.tf_sess.run(optimizer, feed_dict=feed_dict)
            loss, acc = self.tf_sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            if np.isnan(loss):
                loss = np.nan_to_num(loss)
            rates.append(learning_rate)
            t_loss.append(loss)
            t_acc.append(acc)

            print(f'epoch {i + 1}: learning rate = {learning_rate:.10f}, loss = {loss:.10f}')
        if plot_charts:
            iters = np.arange(len(rates))
            plt.title("Learning Rate (log) vs. Iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")
            plt.plot(iters, rates, 'b')
            plt.show()

            plt.plot(rates, t_loss, 'b')
            plt.title("Loss vs. Learning Rate (log)")
            plt.xlabel("Learning Rate")
            plt.ylabel("Loss")
            plt.show()

        # Calculate the learning rate based on the biggest derivative betweeen the loss and learning rate
        dydx = list(np.divide(np.diff(t_loss), np.diff(rates)))
        start = rates[dydx.index(max(dydx))]
        print("Chosen start learning rate:", start)
        print()
        return start

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

    def conv_layer(self, input, input_channels, filters, filter_size) -> tf.nn.relu:
        """
        Creates a convolutional layer for the model to feed into

        :param input: the input for the model layer
        :param input_channels: the number of channels the input contains
        :param filters: the filters for passing into the convolutional layer
        :param filter_size: the number of filters
        :return: tf.nn.relu layer, the constructed convolutional layer
        """

        weights = self.create_weights(shape=[filter_size, filter_size, input_channels, filters])
        biases = self.create_biases(filters)

        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='VALID')
        layer += biases

        layer = tf.nn.relu(layer)

        return layer

    def pool(self, layer: tf.nn.conv2d, ksize: list, strides: list, padding='VALID') -> tf.nn.max_pool:
        """
        Creates a pooling layer for the model

        :param tf.nn.conv2d layer: the layer to feed into the max_pool layer
        :param list ksize: the kernel size of the pooling layer
        :param list strides: the strides for the pooling layer
        :param str padding: the padding to specify for the pooling layer
        :return: tf.nn.max_pool the constructed max pool layer
        """

        return tf.nn.max_pool(layer, ksize=ksize, strides=strides, padding=padding)

    def flatten_layer(self, layer: tf.nn.conv2d) -> tf.nn.conv2d:
        """
        Flattens a convolutional layer
        :param tf.nn.conv2d layer: the convolutional layer to flatten
        :return: tf.nn.conv2d the flattened layer
        """

        shape = layer.get_shape()
        features = shape[1:4].num_elements()
        layer = tf.reshape(layer, [-1, features])

        return layer

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


if __name__ == "__main__":
    """
    Main method for testing the CNN model
    """

    warnings.filterwarnings("ignore")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    print("TensorFlow Version: ", tf.__version__)

    mnist = ImageDataset(type='TF_MNIST')
    n_train = len(mnist.x_train)
    n_valid = len(mnist.x_valid)
    n_test = len(mnist.x_test)
    image_shape = mnist.shape
    n_classes = mnist.num_classes

    epochs = 50
    learning_rate = 1e-3
    enable_session = True
    dynamic_lr = True
    shape = mnist.shape
    num_classes = mnist.num_classes

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Number of validation examples =", n_valid)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)
    print()

    start = datetime.now()
    cnn = CNN(dataset=mnist,
              num_epochs=epochs,
              learning_rate=learning_rate,
              enable_session=enable_session,
              dynamic_lr=dynamic_lr,
              shape=shape,
              num_classes=num_classes)
    if enable_session:
        cnn.train_model(epochs=epochs, limit=8)
        end = datetime.now()
        print("Time taken to build and train the model on " + str(epochs) + " epochs is:", str(end - start))
        cnn.tf_sess.close()
    else:
        end = datetime.now()
        print("Time taken to build the model on " + str(epochs) + " epochs is:", str(end - start))