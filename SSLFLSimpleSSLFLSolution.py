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

from SSLFLSolution import SSLFLSolution


class SSLFLSimpleSSLFLSolution(SSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))


    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def create_pretext_model_dl(self, input_shape, num_classes):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1000, activation='linear', input_dim=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(input_shape, activation='relu'))

    opt = tf.keras.optimizers.RMSprop()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
  
  def get_latent_space(self, data):

    get_3rd_layer_output = K.function([self.final_model.layers[0].input],
                                      [self.final_model.layers[1].output])

    layer_outs = get_3rd_layer_output([data])
    return np.array(layer_outs[0])


  def create(self):
    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      clients_dataX = self.ssl_fl_problem.clients_dataX
      clients_dataY = self.ssl_fl_problem.clients_dataY
      input_shape = self.ssl_fl_problem.input_shape
      num_classes = self.ssl_fl_problem.num_classes

      n_clients = len(clients_dataX)
      model_list = [self.create_pretext_model_dl(input_shape, num_classes) for i in range(n_clients)]
      final_pretext_model = self.create_pretext_model_dl(input_shape, num_classes)
      final_model = self.create_model_dl(input_shape, num_classes)
      
      # n_rounds = 50
      # n_rounds = 10 # BRACIS Experiment
      # n_rounds = 3
      # n_rounds = 1
      # self.name = 'fssl'
      # self.name = 'fssl-noniid'

      self.name = 'centralized'
      n_rounds = 0
      
      for i in range(n_rounds):
        print("Round ", i)
        for m, clientX, clientY in zip(model_list, clients_dataX, clients_dataY):
          m.set_weights(final_pretext_model.get_weights())
          
          m.fit(clientX, clientX, verbose=0)
          
        
        new_weights = []
        for lidx in range(len(final_pretext_model.get_weights())):
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          new_weights.append(avg_weights)

        final_pretext_model.set_weights(new_weights)
        
      new_weights = final_model.get_weights()
      encoder_layer_limit = 2
      for lidx in range(len(final_pretext_model.get_weights())):
        if lidx == encoder_layer_limit:
          break
        
        avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
        new_weights[lidx] = avg_weights

      final_model.set_weights(new_weights)

      categY = tf.keras.utils.to_categorical(self.ssl_fl_problem.trainY, num_classes = num_classes)

      final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=300, verbose=0)
      print(final_model.layers[0].input)
      
      '''
      for i in range(300):
        print("Epoch ", i)
        final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=1)
        server_pretext_models = [final_model, final_pretext_model]
        new_weights = final_model.get_weights()
        for lidx in range(len(final_pretext_model.get_weights())):
          if lidx == encoder_layer_limit:
            break
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          new_weights[lidx] = avg_weights
        
        final_model.set_weights(new_weights)
        
      final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=1)
      '''

      self.final_model = final_model

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