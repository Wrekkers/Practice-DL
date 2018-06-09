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

    Z1 = tf.nn.relu(tf.matmul(X, Wx) + bh)

    #Z2 = tf.nn.relu(tf.matmul(Z1, Wh[0]) + bh[1])

    #Z3 = tf.nn.relu(tf.matmul(Z2, Wh[1]) + bh[2])


    """
    last_layer = tf.nn.relu((tf.matmul(X,Wx) + bh[0]))
    
    
    for i in range(layers-1):
        Z = tf.nn.relu(tf.matmul(last_layer,Wh[i]) + bh[i+1])
        last_layer = Z
    """
    Y = tf.matmul(Z1,Wy) + by
    return Y



def main():
    X, Y, X_test, Y_test = get_data('../fer2013.csv')
     
    D = X.shape[1]  # size of the input layer
    print('Done reading data')
    print(D)
    #D = 48 * 48
    M = 100          # size of hidden layers
    K = 7           # no. of classes
    layers = 2     # no. of hidden layers
    
    expression = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
                     4:'Sad', 5:'Surprise', 6:'Neutral'}
    
    T = one_hot(Y, K)
    
    tfX = tf.placeholder(tf.float32, [None, D])
    tfY = tf.placeholder(tf.float32, [None, K])
    
    Wx = tf.Variable(init_weights([D,M]))
    #bh = tf.Variable(init_weights([layers,M]))
    bh = tf.Variable(init_weights([M]))
    Wh = tf.Variable(init_weights([M,M]))
    #Wh = tf.Variable(init_weights([layers-1,M,M]))
    Wy = tf.Variable(init_weights([M,K]))
    by = tf.Variable(init_weights([K]))
    
    
    y_out = neural_net(tfX, Wx, Wh, Wy, bh, by)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = y_out, labels = tfY))
        
    train_op = tf.train.GradientDescentOptimizer(3*10e-7).minimize(cost)
    
    predict_op = tf.argmax(y_out, axis=1)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    score_old = 0
    score = 1
    for i in range(10000):
        sess.run(train_op, feed_dict = {tfX: X, tfY : T})
        pred = sess.run(predict_op, feed_dict = {tfX: X, tfY : T})
        
        if(i % 10 == 0):
            score = np.mean(pred == Y)
            print('Iteration: ' + str(i) + ', Score: ' + str(score))
            
        if (i % 1000 == 0):
            if (score < score_old):
                break
            else:
                score_old = score


if __name__ == '__main__':
    main()

