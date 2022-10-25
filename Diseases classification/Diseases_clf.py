# -*- coding: utf-8 -*-
"""
Created on Fri May 17 12:19:15 2019

@author: vaibhav pawar
"""

"""
IMPORT 
"""
import os
import glob
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image

from sklearn.model_selection import train_test_split
from tqdm import tqdm
#%matplotlib inline

"""
Read
"""
base_dir = os.path.join('E:/Diseases classification')
train = pd.read_csv('All_Training_70Patient_structured.csv')    # reading the csv file
#train.head()      # printing first five rows of the file
valid = pd.read_csv('All_Validation_10Patient_structured.csv')    # reading the csv file
#valid.head()      # printing first five rows of the file
test = pd.read_csv('All_Testing_20Patient_structured.csv')    # reading the csv file
#test.head()      # printing first five rows of the file
train.columns
