# MNIST Classifier

This repository contains my work as part of my ongoing research efforts with the [Ethos Lab at Stony Brook University](https://github.com/Ethos-lab), supervised by [Prof. Amir Rahmati](https://amir.rahmati.com) and [Pratik Vaishnavi](https://www3.cs.stonybrook.edu/~pvaishnavi/). 

## Dataset

The dataset used for the tasks was the Modified National Institute of Standards and Technology database (MNIST) handwritten digits dataset which can be downloaded by having the import statement `from tensorflow.examples.tutorials.mnist import input_data`

## Python Packages

This code uses TensorFlow version 1.15 and makes use of image libraries, such as `opencv` and `pillow`. Python packages are in the `requirements.txt` file. 

Create a virtual environment and install the needed dependencies using pip
```bash
$ virtualenv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

## Tasks Completed
- Multipurpose data loader with random data augmentation applied on each training batch, which can be found in [`data_loader.py`](https://github.com/asarj/MNIST_Classifier/blob/master/data_loader.py)
- A LeNet architecture-based convolutional neural network classifier, which can be found in [`mnist_cnn_classifier.py`](https://github.com/asarj/MNIST_Classifier/blob/master/mnist_cnn_classifier.py)
- A Mixture Of Experts-based model that utilizes several individual CNNs to improve prediction accuracy which can be found in [`mixture_of_experts.py`](https://github.com/asarj/MNIST_Classifier/blob/master/mixture_of_experts.py)
- Early stopping to prevent overfitting during training
- Learning rate estimator script that uses a Gradient Descent Optimizer to find the optimal learning rate before training
