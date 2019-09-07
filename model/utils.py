from glob import glob
from PIL import Image

import pandas as pd
import numpy as np

import gc

def read_training_im(labels, n, path = './train/'):
    """read_training_im
    
    Read training dataset
    
    Args:
        labels (pandas DF): Pandas dataframe with filename + label
        n (int): Number of images to load (all images are too big)
        path (string): Path of train data
    Returns:
        x_image (nparray): Numpy array of images
        y_label (nparray): Numpy array of labels
    """
    i = 0
    y_label = np.zeros((n))
    x_image = np.zeros((n, 96, 96, 3))
    labels = list(labels.values)
    for row in labels:
        im = Image.open(path + row[0] + '.tif')
        np_im = np.asarray(im)
        x_image[i,:,:,:] = np_im
        y_label[i] = row[1]
        i+=1
        if i >= n:
            break
    x_image = np.asarray(x_image)
    y_label = np.asarray(y_label)
    return(x_image, y_label)

def split_train_val(x_image, y_label, share = 0.9):
    """split_train_val
    
    Split training data into training and validation
    
    Args:
        x_image (nparray): Numpy array of images
        y_label (nparray): Numpy array of labels
        share (float): Share of training data 
    Returns:
        x_train (nparray): Numpy array of training images
        y_train (nparray): Numpy array of training labels
        x_val (nparray): Numpy array of validation images
        y_val (nparray): Numpy array of validation labels
    """
    idx = np.asarray(range(x_image.shape[0]))
    np.random.shuffle(idx)
    x_train = x_image[idx[0:int(x_image.shape[0]*share)]]
    y_train = y_label[idx[0:int(x_image.shape[0]*share)]]
    x_val = x_image[idx[int(x_image.shape[0]*share):x_image.shape[0]]]
    y_val = y_label[idx[int(x_image.shape[0]*share):x_image.shape[0]]]
    return x_train, y_train, x_val, y_val

def normalize_data(x_train, x_val):
    """normalize_data
    
    Normalize the training and validation data
    
    Args:
        x_train (nparray): Numpy array of training images
        x_val (nparray): Numpy array of validation images
    Returns:
        x_train (nparray): Numpy array of normalized training images
        x_val (nparray): Numpy array of normalized validation images
        mean (nparray): Mean of features
        mean (nparray): Standard derivation of features
    """
    x_train = x_train / 255.
    x_val = x_val / 255.
    
    mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(x_train, axis=(0, 1, 2), keepdims=True)
    
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    return(x_train, x_val, mean, std)