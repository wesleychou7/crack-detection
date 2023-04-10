import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from keras.regularizers import l2

class MyModels():
    
    def __init__(self):
        self.models = {}
        self.current_model = None
    
        
    def print_models(self):
        """
        Prints the list of model names.
        """
        print(list(self.models.keys()))
    
    def add_model(self, model_name, model):
        """
        Add a model.
        """
        self.models[model_name] = model
    
    
    def del_model(self, model_name):
        """
        Delete an existing model.
        """
        del self.models[model_name]
    
    
    def set_model(self, model_name):
        """
        Set current model to an existing model.
        """    
        if model_name not in self.models:
            raise ValueError("Invalid model name.")
        
        self.current_model = self.models[model_name]