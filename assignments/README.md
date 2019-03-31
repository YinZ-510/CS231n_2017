## Assignment1

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- develop proficiency in writing efficient **vectorized** code with numpy
- implement and apply a k-Nearest Neighbor (**kNN**) classifier
- implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- implement and apply a **Softmax** classifier
- implement and apply a **Two layer neural network** classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using **higher-level representations** than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

### Q1: k-Nearest Neighbor classifier

The IPython Notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine

The IPython Notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Two-Layer Neural Network

The IPython Notebook **two_layer_net.ipynb** will walk you through the implementation of a two-layer neural network classifier.

### Q5: Higher Level Representations: Image Features

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.


## Assignment2

In this assignment you will practice writing backpropagation code, and training
Neural Networks and Convolutional Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- effectively **cross-validate** and find the best hyperparameters for Neural
  Network architecture
- understand the architecture of **Convolutional Neural Networks** and train
  gain experience with training these models on data

### Q1: Fully-connected Neural Network
The IPython notebook **FullyConnectedNets.ipynb** will introduce you to our
modular layer design, and then use those layers to implement fully-connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization
In the IPython notebook **BatchNormalization.ipynb** you will implement batch
normalization, and use it to train deep fully-connected networks.

### Q3: Dropout
The IPython notebook **Dropout.ipynb** will help you implement Dropout and explore
its effects on model generalization.

### Q4: Convolutional Networks
In the IPython Notebook **ConvolutionalNetworks.ipynb** you will implement several new layers that are commonly used in convolutional networks.

### Q5: PyTorch / TensorFlow on CIFAR-10
For this last part, you will be working in either TensorFlow or PyTorch, two popular and powerful deep learning frameworks. 
Open up either **PyTorch.ipynb** or **TensorFlow.ipynb**. There, you will learn how the framework works, culminating in training a  convolutional network of your own design on CIFAR-10 to get the best performance you can.


```
Anaconda 创建虚拟环境：conda create -n cs231n python=3.6

安装依赖：pip install -r requirements.txt
```
