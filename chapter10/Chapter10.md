# Introduction to Neural Networks with OpenCV
This chapter introduces a family of machine learning models called **artificial neural networks (ANNs)**, or sometimes just **neural networks**. A key characteristic of these models is that they attempt to learn relationships among variables in a multi-layered fashion; they learn multiple functions to predict intermediate results before combining these into a single function to predict something meaningful (such as the class of an object). Recent versions of particular, ANNs with many layers, called **deep neural networks (DNNs)**. We will experiment with both shallower ANNs and DNNs in this chapter.

We have already gained some exposure to machine learning in other chapters - especially in *Chapter 7*, *Building Custom Object Detectors*, wherewe developed a car/non-car classifier using SURF descriptors, a BoW, and an SVM. With this basis for comparison, you might be wondering, what is so special about ANNs? Why are we devoting this book's final chapterto them?

ANNs aim to provide superior accuracy in the following circumstances:
- There are many input variables, which may have complex, nonlinear relationships to each other.
- There are many output variables, which may have complex, nonlinear relationships to the input variables. (Typically, the output variables in a classification problem are the confidence scores for the classes, so if there are many classes, there are many output variables.)
- There are many hidden (unspecified) variables that may have complex, nonlinear relationships to the input and output variables. DNNs even aim to model *multiple* layers of hidden variables, which are interrelated primarily to each other rather than being related primarily to input or output variables.

These circumstances exist in many - perhaps most - real-world problems. Thus, the promised advantages of ANNs and DNNs are enticing. On the other hand, ANNs and especially DNNs are notoriously opaque models, insofar as they work by predicting the existence of an arbitrary number of nameless, hidden variables that may relate to everything else.

Over the course of this chapter, we will cover the following topics:
- Understanding ANNs as a statistical model and as a toll for supervised machine learning.
- Understanding ANN topology or, in other words, the organization of an ANN into layers of interconnected neurons. Particularly, we will consider the topology that enables an ANN to act as a type of classifier known as a **multi-layer perceptron (MLP)**.
- Training and using ANNs as classifiers in OpenCV.
- Building an application that detects and recognizes handwritten digits (0 to 9). For this, we will train an ANN based on a widely used dataset called MNIST, which contains samples of handwritten digits.
- Loading and using-pretrained DNNs in OpenCV. We will cover examples of object classification, face detection, and gender classification with DNNs.

By the end of this chapter, you will be in a good position to train and use ANNs in OpenCV, to use pre-trained DNNs from a variety of sources, and to start exploring other libraries that allow you to train your own DNNs.

## Technical requirements
This chapter uses Python, OpenCV, and Numpy. Please refer to ```Chapter 1```, *Setting Up OpenCV*, for installation instructions.

The completed code and sample videos for this chapter can be found in this book's [GitHub repository](https://github.com/PacktPublishing/Learning-OpenCV-4-Computer-Vision-with-Python-Third-Edition), in the ```chapter10``` folder.

## Understanding ANNs
Let's define ANNs in terms of their basic role and components. Although much of the literature on ANNs emphasizes the idea that they are *biologically inspired* by the way neurons connect in a brain, we don't need to be biologists or neuroscientists to understand the fundamental concepts of an ANN.

First of all, an ANN is a **statistical model**. What is a statistical model? A statistical model is a pair of elements, namely the space ```S```(a set of observations) and the probability, ```P```, where ```P``` is a distribution that approsimates ```S```(in other words, a function that would generate a set of observations that is very similar to ```S```).

Here are two different ways of think of ```P```:
- ```P``` is a simplification of a complex scenario.
- ```P``` is the function that generated ```S```  in the first place, or at the very least a set of observations very similar to ```S```.

Thus, ANNs are models that take a complex reality, simplify it, and deduce a function to(approximately) represent the statistical observations we would expect from that reality, in a mathematical form.

ANNs, like other types of machine learning models, can learn from observations in one of the following ways:
- **Supervised learning**: Under this approach, we want the model's training process to produce a function that maps a known set of input variables to a kwnown set of output variables. We know, *a priori*, the nature of the prediction problem, and we delegate the process of finding a function that solves this problem to the ANN. To train the model, we must provide input samples along with the correct, corresponding outputs. For a classificaton problem, the output variables may be confidence scores for one or more classes.
- **Unsupervised learning**: Under this approach, the set of output variables is not known *a priori*. The model's training process must yield a set of output variables, as well as a function to map the input variables to these output variables. For a classification problem, unsupervised learning can lead to the discovery of previously unknown classes, such as previously unknown diseases in the context of mecical data. Unsupervised learning may use techniques including(but not limited to) clustering, which we explored in the context of BoW models in ```Chapter 7```, *Building Custom Object Detectors*.
- **Reinforcement learning**: This approach turns the typical prediction problem upside down. Before training the model, we already have a system that yields values for a known set of output variables when we feed it values for a known set of input variables. We know, *a priori*, a way of scoring a sequence of outputs based on thei goodness(desirability) or lack thereof. However, we might not know the real function that maps inputs to outputs - or, even if we do know it, it is so complex that we connot solve it for optimal inputs. Thus,we want the model's training process to produce a function that predicts the next-in-sequence optimal inputs, based on the last outputs. During training, the model learns form the score that eventually arises from its actions(its chosen inputs). Essentially, the model must learn to become a good decision-maker within the context of a particular system of rewards and punishments.

Throughout the remainder of this chapter, we will confine our discussions to supervised learning, as this is the most common approach to machine learning in the context of computer vision.

The next step in our journey toward comprehending ANNs is to understand how an ANN improves on the concept of a simple statistical model, and on other types of machine learning.

What if the function that generated the dataset is likely to take a large number of **neuron**, **nodes**, or **units**, each of which is capable of approximating the function that created the inputs. In mathematics, approximation is the process of defining a simpler function whose output is similar to that of a more complex function, at least for some range of inputs.

The difference between the approximate function's output and the original function's output is called the **error**. A defining characteristic of a neural network is that ther neurons must be capable of approximating a nonlinear function.

Let's take a closer look at neurons.

## Understanding neurons and perceptrons
Often, to solve a classification problem, an ANN is designed as a **multi-layer perceptron(MLP)**, in which each neuron acts as a kind of binary classifier called a **perceptron**. The perceptron is a concept that dates back to the 1950s. To put it simply, a perceptron is a function that takes a number of inputs and produces a single value. Each of the inputs has an associated weight that signifies its importance in an **activation function**. The activation function should have a nonlinear response; for example, a sigmoid function(sometimes called an S-curve) is a common choice. A threshold function, called a **discriminant**, is applied to the activation function's output to convert it into a binary classification of 0 or 1. Here is a visualization of this sequence, with inputs on the left, the activation function in the middle, and the discriminant on the right.

What do the input weights represent, and how are they determined?

Neurons are interconnected, insofar as one neuron's output can be an input for many other neurons. Each input weight defines the strength of the connection between two neurons. These weights are adaptive, meaning that they change in time according to a learning algorithm.

Due to the neurons' interconnectedness, the network has layers. Now let's examine how these layers are typically organized.