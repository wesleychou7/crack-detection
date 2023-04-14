import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from PIL import Image, ImageOps

def create_model(input_shape):
    
    tf.random.set_seed(42)
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


def load_images(directory, image_size):
    """
    Loads images from given directory. The images are resized to the specified 
    size.

    Args:
        directory (string)
        image_size (tuple)

    Returns:
        List of images (PIL Images array)
    """
    
    images = []

    filenames = os.listdir(directory)
    
    for filename in filenames:
        
        if filename.endswith(".jpg"):
                
            img = Image.open(os.path.join(directory, filename))
            
            # resize image to specified size
            img = ImageOps.fit(img, image_size, method=Image.LANCZOS)
            
            images.append(img)
    
    return images


def grayscale(images):
    """
    Grayscales array of PIL images.

    Args:
        images (PIL Images array)
    """
    gs_images = []
    
    for img in images:
        # grayscale image
        gs_images.append(ImageOps.grayscale(img))
        
    return gs_images