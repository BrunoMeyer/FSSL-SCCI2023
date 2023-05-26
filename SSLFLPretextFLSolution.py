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

import tensorflow_addons as tfa


class SupervisedContrastiveLoss(tf.keras.losses.Loss):
  def __init__(self, temperature=1, name=None):
    super().__init__(name=name)
    self.temperature = temperature

  def __call__(self, labels, feature_vectors, sample_weight=None):
    # Normalize feature vectors
    feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
    # Compute logits
    logits = tf.divide(
        tf.matmul(
            feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
        ),
        self.temperature,
    )
    return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

class SSLFLPretextFLSolution(SSLFLSolution):
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
    # model.add(tf.keras.layers.Dense(1000, activation='linear', input_dim=input_shape))
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))
    # model.add(tf.keras.layers.Dropout(0.2))
    
    # model.add(tf.keras.layers.Dense(input_shape, activation='relu'))
    # model.add(tf.keras.layers.Dense(len(self.ssl_fl_problem.clients_pretext_dataY[0][0]), activation='relu'))
    # model.add(tf.keras.layers.Dense(65535, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(1026, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(1025, activation='softmax'))
    # model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='linear'))

    opt = tf.keras.optimizers.RMSprop()
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)

    # model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    model.compile(optimizer=opt, loss=SupervisedContrastiveLoss(1), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # model.compile(optimizer=opt, loss=SupervisedContrastiveLoss(0.5), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    # model.compile(optimizer=opt, loss=tf.keras.metrics.SparseCategoricalAccuracy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return model
  
  def get_latent_space(self, data):

    get_3rd_layer_output = K.function([self.final_model.layers[0].input],
                                      [self.final_model.layers[1].output])

    layer_outs = get_3rd_layer_output([data])
    return np.array(layer_outs[0])


  def create(self, n_rounds=0, name='centralized'):
    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      clients_dataX = self.ssl_fl_problem.clients_dataX
      clients_dataY = self.ssl_fl_problem.clients_dataY
      input_shape = self.ssl_fl_problem.input_shape
      num_classes = self.ssl_fl_problem.num_classes
      clients_pretext_dataY = self.ssl_fl_problem.clients_pretext_dataY

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

      self.name = name
      if name == 'centralized':
        n_rounds = 0
      
      
      for i in range(n_rounds):
        updated_models = []
        print("Round ", i)
        for m, clientX, pretext_clientY in zip(model_list, clients_dataX, clients_pretext_dataY):
          m.set_weights(final_pretext_model.get_weights())
          
          # m.fit(clientX, clientX, verbose=0)
          # m.fit(clientX, pretext_clientY)
          # pretext_clientY = np.array(pretext_clientY)
          # pretext_clientY[pretext_clientY >= 1025] = 1025

          # pretext_clientY = tf.keras.utils.to_categorical(pretext_clientY, num_classes = 1026)
          # # pretext_clientY = tf.keras.utils.to_categorical(pretext_clientY, num_classes = 65535)
          # pretext_clientY = np.sum(pretext_clientY, axis=1)
          
          # pretext_clientY = tf.keras.utils.to_categorical(pretext_clientY[:, 0], num_classes = 1026)
          # m.fit(clientX, pretext_clientY, batch_size=128)

          pretext_clientY = np.array(pretext_clientY)
          pretext_clientX = np.array(clientX[np.where(pretext_clientY[:, 0] < 1025)[0], :])
          # pretext_clientY = pretext_clientY[np.where(pretext_clientY[:, 0] < 1025)[0]][:, 0]
          pretext_clientY = pretext_clientY[np.where(pretext_clientY[:, 0] < 1025)[0]][:, 0].reshape((-1, 1))
          # pretext_clientY = tf.keras.utils.to_categorical(pretext_clientY[:, 0], num_classes = 1025)
          # pretext_clientY[pretext_clientY > 100] = 1
          # pretext_clientY[pretext_clientY <= 100] = 0
          # m.fit(pretext_clientX, pretext_clientY, batch_size=1)
          try:
            m.fit(pretext_clientX, pretext_clientY)
            updated_models.append(m)
          except:
            print("Fail")


          
          # print(pretext_clientY[0])
          # print(pretext_clientY[0].dtype)
          # print(pretext_clientY[0].shape)
          # m.fit(clientX, pretext_clientY)
        
        new_weights = []
        for lidx in range(len(final_pretext_model.get_weights())):
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in updated_models])/len(updated_models)
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