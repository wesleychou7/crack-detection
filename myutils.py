import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

def create_model(size):
    
    tf.random.set_seed(42)
    input_shape = (size[0], size[1], 1)

    model = Sequential(
        [
            Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2,2)),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D((2,2)),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')
            
        ], name = "mymodel"
    )
    
    return model