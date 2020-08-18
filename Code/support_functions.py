# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 13:38:32 2019
@author: Carmen Martinez Barbosa
"""

import numpy as np
import pandas as pd
#from skopt import BayesSearchCV
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import CategoricalDtype
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import rasterio

class DummyEncoder(BaseEstimator, TransformerMixin):
    """
    Class to make dummy encoding.
    Based on: https://gist.github.com/psinger/ef4592492dc8edf101130f0bf32f5ff9
    NOTE: REVISE THE TRANSFORM. WHAT IF TEST DATA HAS != OR LESS CATEGORIES?
    """
    def __init__(self, min_frequency=1, dummy_na= True, cols_for_encoding= []):
        self.min_frequency = min_frequency
        self.dummy_na = dummy_na
        self.cols_for_enc= cols_for_encoding
        self.categories = dict()
        self.features = []

    def fit(self, X, y=None):
        'Here the categories are built'
        for col in self.cols_for_enc:
            counts = pd.value_counts(X[col])
            self.categories[col] = list(set(counts[counts >= self.min_frequency].index.tolist()))
        return self

    def transform(self, X, y=None):
        for col in self.cols_for_enc:
            X = X.astype({col: CategoricalDtype(self.categories[col], ordered=True)})
        ret = pd.get_dummies(X[self.cols_for_enc], dummy_na=self.dummy_na)
        self.features = ret.columns
        other_feat= list(set(X.columns)-set(self.cols_for_enc))
        ret= pd.merge(X[other_feat], ret, left_index= True, right_index= True)
        return ret

    def get_feature_names(self):
        return self.features

class StandardScalerOnColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform scaling on.
    """
    from sklearn.preprocessing import StandardScaler

    def __init__(self, columns = []):
        self.columns = columns
        self.s= StandardScaler()

    def fit(self, X, y=None):
        self.s.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        X1= X.copy()
        X1[self.columns] = self.s.transform(X1[self.columns])
        return X1

    def inverse_transform(self, X, y=None):
        X1 = X.copy()
        X1[self.columns] = self.s.inverse_transform(X[self.columns])
        return X1

class MinMaxScalerOnColumns(BaseEstimator, TransformerMixin):
    """
    Transformer to select a single column from the data frame to perform scaling on.
    """
    from sklearn.preprocessing import MinMaxScaler

    def __init__(self, columns = [], feature_range= (0,1)):
        self.columns = columns
        self.range = feature_range
        self.s= MinMaxScaler(feature_range= self.range)

    def fit(self, X, y=None):
        self.s.fit(X[self.columns])
        return self

    def transform(self, X, y=None):
        X1= X.copy()
        X1[self.columns] =self.s.transform(X[self.columns])
        return X1

    def inverse_transform(self, X, y=None):
        X1 = X.copy()
        X1[self.columns] = self.s.inverse_transform(X[self.columns])
        return X1




def create_dense_network(num_inputs= 3,
                         layer_sizes= [],
                         activation='softmax',
                         loss = 'categorical_crossentropy',
                         optimizer ='adam',
                         dropout= 0.2,
                         metrics=None):
    """
    Creates a keras Model for a simple Dense network
    :param num_inputs: number of inputs to the first layer
    :param layer_sizes: list with the size of the hidden and output layers,
                        length-1 of this list corresponds to the number of hidden layers that are included in the network.
                        the last term of the list corresponds to the output layer and the value
                        is the number of outputs the network will provide. For regression, this value = 1.
    :param activation: activation function to use, see https://keras.io/activations/
    :param loss: objective function to be used, see https://keras.io/losses/
    :param optimizer: optimizer to be used, see https://keras.io/optimizers/
    :param metrics: list of metrics to be evaluated by the model during training and testing
    :return: a keras Model object, see https://keras.io/models/model/
    """
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Dropout

    if metrics is None:
        metrics = ['accuracy']

    model = Sequential()

    model.add(Dense(input_dim=num_inputs, units=layer_sizes[0], activation=activation))

    for layer_size in layer_sizes[1:-1]:
        model.add(Dense(units=layer_size, activation=activation))
        model.add(Dropout(dropout))

    model.add(Dense(units=layer_sizes[-1], activation= activation))

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    return model


def plot_history(model_history, filename= None):
    #r'P:/11203770-006-ls4ri/LSM/models/history_ann.png'
    fig, (ax1, ax2)= plt.subplots(1,2, figsize= (10, 5))
    if 'acc' in model_history.history.keys():
        ax1.plot(model_history.history['acc'])
        ax1.plot(model_history.history['val_acc'])
        ax1.set_title('model acc')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='best')

        ax2.plot(model_history.history['loss'])
        ax2.plot(model_history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='best')

    if 'accuracy' in model_history.history.keys():
        ax1.plot(model_history.history['accuracy'])
        ax1.plot(model_history.history['val_accuracy'])
        ax1.set_title('model acc')
        ax1.set_ylabel('accuracy')
        ax1.set_xlabel('epoch')
        ax1.legend(['train', 'test'], loc='best')

        ax2.plot(model_history.history['loss'])
        ax2.plot(model_history.history['val_loss'])
        ax2.set_title('model loss')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('epoch')
        ax2.legend(['train', 'test'], loc='best')

    if filename is not None:
        plt.savefig(filename , bbox= 'tight')



def visualize_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          save_file= False,
                          plotfilename= 'confusion_matrix.png'
                          ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input:  cm -> confission matrix
            classes -> Array with strings indicating each of the classes in target. Ex: ['0', '1'] for a binary target.
            title -> title of the plot
            cmap -> color map
    Output: the confusion matrix plot
    """
    import itertools
    from matplotlib import pyplot as plt
    import numpy as np

    cmap=plt.cm.Blues
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    #plt.xticks(tick_marks, classes, rotation=45)
    #plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if (save_file):
        plt.savefig(plotfilename, bbox= 'tight')
    plt.show()

def pooling(array, size= (2,2), ptype= 'max'):
    """
    Function to make a pooling
    Inputs: array
            size: Size of the pooling
            type: 'max', 'min', 'avg'
    Outputs: A pooled array
    """

    POOL= []

    for i in range(0, array.shape[0], size[0]): # over rows of array
        reduced = []
        for j in range(0, array.shape[1], size[1]):
            matrix= array[i:i+2, j:j+2]
            if (ptype== 'max'):
                reduced.append(max(mat.flatten()))
            if (ptype== 'min'):
                reduced.append(min(mat.flatten()))
            if (ptype== 'avg'):
                reduced.append(np.mean(mat.flatten()))
        POOL.append(reduced)
    return POOL
