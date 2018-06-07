#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:12:16 2018

@author: bh387886
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def view_data(X,Y):
    
    ####
    expression = {0:'Angry', 1:'Disgust', 2:'Fear', 3:'Happy',
                     4:'Sad', 5:'Surprise', 6:'Neutral'}
    i = 0
    for img in X:
        plt.imshow(img.reshape(48, 48), cmap='gray')
        plt.title(expression[Y[i]])
        plt.show()
        i += 1
        prompt = input('Enter (y/n) :')
        if (prompt.lower() == 'y' or i > 10):
            break
    ####
    

def get_data(filename):
    
    ####
    raw = open(filename)
    
    first = True
    
    Y_train = []
    Y_test = []
    X_train = []
    X_test = []
    
    for line in raw:
        if (first):
            first = False
        
        else :
            row = line.split(',')
            if (row[2].rstrip() == 'Training' or row[2].rstrip() == 'PublicTest'):
                Y_train.append(int(row[0]))
                X_train.append([int (pixel_val) for pixel_val in row[1].split(' ')])
            if (row[2].rstrip() == 'PrivateTest'):
                Y_test.append(int(row[0]))
                X_test.append([int (pixel_val) for pixel_val in row[1].split(' ')])

    
    ####
    X_train, X_test = np.array(X_train)/255.0, np.array(X_test) / 255.0
    Y_train, Y_test = np.array(Y_train) , np.array(Y_test)
    
    return X_train, Y_train, X_test, Y_test


def main():
    
    filename = '../fer2013.csv'
    #train_set,test_set,score_set = get_data(filename) 
    
    X,Y,X_t,Y_t = get_data(filename) 
    #view_data(X, Y)
    
    
if __name__ == '__main__':
    main()