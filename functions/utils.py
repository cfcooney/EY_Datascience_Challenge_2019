"""
Name: Ciaran Cooney
Date: 05/04/2019
Description: Some utility functions used in the project, seen elsewhere in the notebooks.
"""
import pandas as pd  
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score
from keras import backend as K

def accuracy_f_score(y_pred,y_true):
    print(f"Accuracy score: {round(accuracy_score(y_true, y_pred) * 100,2)}%")
    print(f"F1 score: {round(f1_score(y_true, y_pred) * 100,3)}%")
    

def in_city(x_pred,y_pred):
    
    if (3750901.5068 <= x_pred <= 3770901.5069) and (-19268905.6133 <= y_pred <= -19208905.6133):
        return 1
    else:
        return 0

def journey_time(x,y):
    """
    Compute journey time in seconds.
    """
    x = pd.to_datetime(x)
    y = pd.to_datetime(y)
    return (y-x).total_seconds()

def sigmoid(x):
    e = np.exp(1)
    y = 1/(1+e**(-x))
    return y

def to_binary(x):
    result = []
    for n in x:
        result.append(np.argmax(n))
    return result

def sign(x):
    if np.sign(x) == 1:
        return 0
    else:
        return 1

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
    
    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
