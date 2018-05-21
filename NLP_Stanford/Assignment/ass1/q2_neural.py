#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
#from q2_gradcheck import gradcheck_naive


def forward_backward_prop(X, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Note: compute cost based on `sum` not `mean`.
    ### YOUR CODE HERE: forward propagation
    #X= X.reshape((X.shape[0], 1))

    #print ("W1 :%s", W1.shape)
    #print ("W2 :%s", W2.shape)
    
    z1 = (np.dot(X,W1)) + b1
    h = sigmoid(z1)
    #print ("h :%s", h.shape)
    z2 = (np.dot(h,W2)) + b2
    yhat = softmax(z2)

    prob = np.log(softmax(yhat*X))
    cost = -1 * np.sum(np.sum(labels*prob))

    del1 = yhat - labels
    gradb2 = del1
    gradW2 = np.dot(h.T,del1)
    #print ("gradW2 :%s", gradW2.shape)
    #print ("del1 :%s", del1.shape)
    del2 = np.dot(del1,W2.T)
    #print ("del2 :%s", del2.shape)
    
    
    del3 = del2*sigmoid_grad(z1)
    #print ("del3 :%s", del3.shape)
    
    gradb1 = del3
    gradW1 = np.dot(X.T,del3)
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 2
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    forward_backward_prop(data, labels, params, dimensions)
    #gradcheck_naive(lambda params:
    #   forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print ("Running your sanity checks...")
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    #your_sanity_checks()
