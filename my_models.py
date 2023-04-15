import matplotlib.pyplot as plt
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
        
    
    def get_model(self, model_name):
        """
        Returns a copy of the specified model.
        """
        return self.models[model_name]
    
    
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
        
    
    def del_all_models(self):
        """
        Deletes all models.
        """
        self.models = {}
    
    
    def set_model(self, model_name):
        """
        Set current model to an existing model.
        """    
        if model_name not in self.models:
            raise ValueError("Invalid model name.")
        
        self.current_model = self.models[model_name]
    
    
    def run_and_evaluate(self, X_train, y_train, X_cv, y_cv, learning_rate, epochs):
        """
        Run all the models and show its train and validation accuracy plots.
        """
        fig, axes = plt.subplots(ncols=len(self.models), figsize=(12,3), sharey=True)
        fig.supylabel('Accuracy')
        fig.supxlabel('Epoch')
                
        for i, (model_name, model) in enumerate(self.models.items()):
            
            model.compile(
                loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy']
            )
            
            history = model.fit(X_train, y_train, epochs=epochs, 
                                validation_data=(X_cv, y_cv), verbose=0)
            
            axes[i].set_title(model_name)
            axes[i].plot(history.history['accuracy'], label='accuracy')
            axes[i].plot(history.history['val_accuracy'], label = 'val_accuracy')
            if i==0: axes[i].legend()
            
            cv_loss, cv_acc = model.evaluate(X_cv, y_cv, verbose=0)
            print(f"{model_name}:")
            print(f"\tloss = {cv_loss}\n\taccuracy = {cv_acc}")
            
            