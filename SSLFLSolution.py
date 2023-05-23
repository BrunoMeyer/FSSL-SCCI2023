import pandas as pd
import numpy as np
import argparse


from cybersecurity_datasets import load_nslkdd, load_toniot, load_botiot

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K


from fed_bench_utils import partition_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

from sklearn.manifold import TSNE

import imblearn
# from imblearn.under_sampling import RandomUnderSampler

import matplotlib.pyplot as plt

import collections


class SSLFLSolution(object):
  def __init__(self, ssl_fl_problem):
    self.ssl_fl_problem = ssl_fl_problem
    self.final_model = None

  def report_metrics(self):
    self.ssl_fl_problem.report_metrics(self)

  def predict(self, X):
    return self.final_model.predict(X)
    
  def report_metrics_on_cross_val(self, **kargs):
    self.ssl_fl_problem.report_metrics_on_cross_val(self, **kargs)

  def plot_tsne_on_cross_val(self, n_folds = 10):
    self.ssl_fl_problem.plot_tsne_on_cross_val(self, n_folds=n_folds)

class SimpleFLSolution(SSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='relu', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # opt = tf.keras.optimizers.Adam(learning_rate=0.1)
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


  def get_latent_space(self, data):

    inp = self.final_model.input                                           # input placeholder
    outputs = [layer.output for layer in self.final_model.layers]          # all layer outputs
    functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions

    # Testing
    # test = np.random.random(input_shape)[np.newaxis,...]
    for x in data:
      layer_outs = [func(data) for func in functors]
      print(layer_outs[0].shape)
      exit()
      
  def create(self):
    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      clients_dataX = self.ssl_fl_problem.clients_dataX
      clients_dataY = self.ssl_fl_problem.clients_dataY
      input_shape = self.ssl_fl_problem.input_shape
      num_classes = self.ssl_fl_problem.num_classes

      n_clients = len(clients_dataX)
      model_list = [self.create_model_dl(input_shape, num_classes) for i in range(n_clients)]
      final_model = self.create_model_dl(input_shape, num_classes)
      
      n_rounds = 100
      # n_rounds = 1
      
      for i in range(n_rounds):
        for m, clientX, clientY in zip(model_list, clients_dataX, clients_dataY):
          m.set_weights(final_model.get_weights())
          
          categY = tf.keras.utils.to_categorical(clientY, num_classes = num_classes)
          m.fit(clientX, categY)
          
        new_weights = []
        for lidx in range(len(final_model.get_weights())):
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          new_weights.append(avg_weights)

        final_model.set_weights(new_weights)

      self.final_model = final_model

  def predict(self, X):
    y_pred = np.argmax(self.final_model.predict(X), axis=1)
    return y_pred




class SimpleNonFLSolution(SSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='relu', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    opt = tf.keras.optimizers.SGD(learning_rate=0.1)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def create_model(self, input_shape, num_classes):
    return RandomForestClassifier(max_depth=2, random_state=0)

  def create(self):
    input_shape = self.ssl_fl_problem.input_shape
    num_classes = self.ssl_fl_problem.num_classes


    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      
      # clf = RandomForestClassifier(max_depth=2, random_state=0)
      # clf = self.create_model(input_shape, num_classes)
      # clf.fit(self.ssl_fl_problem.trainX, self.ssl_fl_problem.trainY)
      
      clf = self.create_model_dl(input_shape, num_classes)
      categY = tf.keras.utils.to_categorical(self.ssl_fl_problem.trainY, num_classes = num_classes)
      clf.fit(self.ssl_fl_problem.trainX, categY, epochs=100)

      self.final_model = clf

  def predict(self, X):
    y_pred = np.argmax(self.final_model.predict(X), axis=1)
    return y_pred
       


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()


  

  

if __name__ == "__main__":
  main()