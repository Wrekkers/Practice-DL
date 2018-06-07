# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:43:03 2018

@author: BHANU
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_in_copy import get_data

"""
This function creates the one hot encoding 
of the output vector of size N,K
where N is the sample size and K is the no. of classes.
"""
def one_hot(Y,K):
    T = np.zeros(shape = (len(Y),K))
    for i in range(len(Y)):
        T[i, Y[i]]=1
    return T

def init_weights(shape):
    return tf.random_normal(shape, stddev = 0.01)


def neural_net(X, Wx, Wh, Wy, bh, by):
    
    layers = Wh.shape[0]
    
    last_layer = tf.nn.relu((tf.matmul(X,Wx) + bh[0]))
    
    
    for i in range(layers-1):
        
    
    raise NotImplementedError

def main():
    #X, Y, X_test, Y_test = get_data('../fer2013.csv')
     
    #D = X.shape[1]  # size of the input layer
    D = 48 * 48
    M = 100          # size of hidden layers
    K = 7           # no. of classes
    layers = 10     # no. of hidden layers
    
    expression = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
                     4:'Sad', 5:'Surprise', 6:'Neutral'}
    
    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])
    
    Wx = tf.Variable(init_weights([D,M]))
    bh = tf.Variable(init_weights([layers-1,M]))
    Wh = tf.Variable(init_weights([layers-1,M,M]))
    Wy = tf.Variable(init_weights([M,K]))
    by = tf.Variable(init_weights([K]))
    
    
    
    print(W[0])
    

if __name__ == '__main__':
    main()

