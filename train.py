# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 19:30:07 2017

@author: setty
"""
from utils import *
import time
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def features_extract(imagename_list):
    cars = imagename_list[0]
    notcars = imagename_list[1]

    t=time.time()
    #n_samples = 2000
    #random_idxs = np.random.randint(0, len(cars), n_samples)

    #test_cars = np.array(cars)[random_idxs]
    #test_notcars = np.array(notcars)[random_idxs]
    
    test_cars = np.array(cars)
    test_notcars = np.array(notcars)

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()

    parameters = {'color_space': 'YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                  'orient': 9,  # Number of orientations for HOG features
                  'pix_per_cell': 8,  # Number of pixels per cell in HOG features
                  'cell_per_block': 2,  # Number of cells per block in HOG features
                  'hog_channel': "ALL",  # HOG features to be extracted from which channel
                  'spatial_size': (16, 16),  # Size of the spatial features
                  'hist_bins': 16,  # Number of color histogram bins
                  'spatial_feat': True,  # If spatial features should be included
                  'hist_feat': True,  # If color features should be included
                  'hog_feat': True  # If HOG features should be included
                  }

    car_features = extract_features(test_cars, cspace=color_space,spatial_size=spatial_size,
                        hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)


    notcar_features = extract_features(test_notcars, cspace=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

    print(time.time() -t ,'seconds to compute features')
    
    return car_features, notcar_features, parameters


def train(car_features,notcar_features,parameters):
    
    # Verical stacking of car features and notcar features
    X = np.vstack((car_features,notcar_features)).astype(np.float64)

    # Compute the mean and std of stacked features to be used for later scaling.
    X_scaler = StandardScaler().fit(X)

    # Use mean and std computed to perform standardization by centering and scaling
    scaled_X = X_scaler.transform(X)

    # Labeling '1' for cars features and '0' for notcar features
    y = np.hstack((np.ones(len(car_features)),np.zeros(len(notcar_features))))

    # randomize a selection for training(80%) and testing(20%)
    rand_state = np.random.randint(0,100)
    X_train,X_test,y_train,y_test = train_test_split(scaled_X,y,test_size =0.20,random_state = rand_state)

    # Linear Support Vector Classifier 
    svc = LinearSVC()

    t = time.time()

    # fit to the training data , returning a "best fit" hyperplane that divides, or categorizes, 
    # the  data and the learning is done.
    svc.fit(X_train,y_train)

    print(round(time.time() -t, 2),'seconds to train SVC')

    print("test accuracy of SVC", round(svc.score(X_test,y_test),4))
    
    return svc, X_scaler, parameters