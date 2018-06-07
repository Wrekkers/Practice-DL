#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 12:12:16 2018

@author: bh387886
"""

import numpy as np
import pandas as pd



def view_data():
    
    ####
    raise NotImplementedError
    ####
    

def get_data(filename):
    
    ####
    raw = open(filename)
    
    first = True
    
    Y_train = []
    Y_test = []
    Y_score = []
    X_train = []
    X_test = []
    X_score = []
    
    for line in raw:
        if (first):
            first = False
        
        else :
            row = line.split(',')
            if (row[2].rstrip() == 'Training'):
                Y_train.append(int(row[0]))
                X_train.append([int (pixel_val) for pixel_val in row[1].split(' ')])
            if (row[2].rstrip() == 'PublicTest'):
                Y_test.append(int(row[0]))
                X_test.append([int (pixel_val) for pixel_val in row[1].split(' ')])
            if (row[2].rstrip() == 'PrivateTest'):
                Y_score.append(int(row[0]))
                X_score.append([int (pixel_val) for pixel_val in row[1].split(' ')])
    
    ####
    
    return 0


def main():
    
    filename = 'fer2013.csv'
    #train_set,test_set,score_set = get_data(filename) 
    
    get_data(filename) 
    
if __name__ == '__main__':
    main()