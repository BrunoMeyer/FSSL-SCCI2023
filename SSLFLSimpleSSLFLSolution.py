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
import pickle

tf.random.set_seed(
    0
)


class SSLFLSimpleSSLFLSolution(SSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    
    
    model = tf.keras.models.Sequential()
    
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape, trainable=False))
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01),  input_dim=input_shape))


    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def create_pretext_model_dl(self, input_shape, num_classes):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(1000, activation='linear', input_dim=input_shape))
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01), input_dim=input_shape))

    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(input_shape, activation='relu'))

    opt = tf.keras.optimizers.RMSprop()
    # opt = tf.keras.optimizers.Adam()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
  
  def get_latent_space(self, data):

    get_3rd_layer_output = K.function([self.final_model.layers[0].input],
                                      [self.final_model.layers[1].output])

    layer_outs = get_3rd_layer_output([data])
    return np.array(layer_outs[0])


  def create(
    self,
    n_rounds=0,
    name='centralized',
    epochs_client=1,
    epochs_server=300,
    verbose_client=0,
    verbose_server=1,
    verbose=1):
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

      self.name = name
      if name == 'centralized':
        n_rounds = 0
      

      log_autoencoder_weights_init_rounds = []
      log_autoencoder_weights_final_rounds = []
      log_rounds_autoencoder = []
      log_rounds_final_model = [] # TODO: Add option to test final model at each round

      log_final_model = None
      eval_reconstruct_train_list = []
      eval_reconstruct_test_list = []
      
      clients_per_round = 5
      for i in range(n_rounds):
        log_round_autoencoder = []
        print("Round ", i)
        round_clients_id = np.arange(len(clients_dataX))
        np.random.shuffle(round_clients_id)

        # for client_id, (m, clientX, clientY) in enumerate(zip(model_list, clients_dataX, clients_dataY)):
        for j, client_id in enumerate(round_clients_id[:clients_per_round]):

          print("Client {} (real client {})".format(j, client_id))
          m = model_list[client_id]
          clientX = clients_dataX[client_id]
          clientY = clients_dataY

          m.set_weights(final_pretext_model.get_weights())
          
          # m.fit(clientX, clientX, verbose=0)
          log = m.fit(clientX, clientX, epochs=epochs_client, verbose=verbose_client)

          log_round_autoencoder.append(log.history)
        
        new_weights = []
        for lidx in range(len(final_pretext_model.get_weights())):
          # avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          avg_weights = sum([np.array(model_list[x].get_weights()[lidx]) for x in round_clients_id[:clients_per_round]])/clients_per_round
          new_weights.append(avg_weights)

        final_pretext_model.set_weights(new_weights)
        log_rounds_autoencoder.append(log_round_autoencoder)


        eval_reconstruct_train = final_pretext_model.evaluate(self.ssl_fl_problem.trainX, self.ssl_fl_problem.trainX)
        eval_reconstruct_test = final_pretext_model.evaluate(self.ssl_fl_problem.testX, self.ssl_fl_problem.testX)
        eval_reconstruct_train_list.append(eval_reconstruct_train)
        eval_reconstruct_test_list.append(eval_reconstruct_test)

      print(eval_reconstruct_train_list)
      print(eval_reconstruct_test_list)

      new_weights = final_model.get_weights()
      encoder_layer_limit = 2
      for lidx in range(len(final_pretext_model.get_weights())):
        if lidx == encoder_layer_limit:
          break
        
        avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
        new_weights[lidx] = avg_weights

      final_model.set_weights(new_weights)

      categY = tf.keras.utils.to_categorical(self.ssl_fl_problem.trainY, num_classes = num_classes)

      a = final_model.get_weights()
      log_final_model = final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=epochs_server, verbose=verbose_server)
      # print(np.sum(a[0]-final_model.get_weights()[0]))
      # print(np.sum(a[1]-final_model.get_weights()[1]))
      # print(np.sum(a[2]-final_model.get_weights()[2]))
      # print(final_model.layers[0].input)
      
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
      log_final_model_weights = self.final_model.get_weights()

      return {
        'final_model_weights': log_final_model_weights,
        'log_autoencoder_weights': log_autoencoder_weights,
        'log_rounds_autoencoder': log_rounds_autoencoder,
        'log_rounds_final_model': log_rounds_final_model,
        'log_final_model': log_final_model.history,
        'eval_reconstruct_train': eval_reconstruct_train_list,
        'eval_reconstruct_test': eval_reconstruct_test_list,
      }

  def predict(self, X):
    y_pred = np.argmax(self.final_model.predict(X), axis=1)
    return y_pred


class SSLFLFreezeKTSSLFLSolution(SSLFLSimpleSSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSimpleSSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    
    
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape, trainable=False))
    
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', kernel_regularizer=tf.keras.regularizers.l1_l2(0.01), input_dim=input_shape, trainable=False))
    # model.add(tf.keras.layers.Dense(128, activation='sigmoid', input_dim=input_shape, trainable=False))
    
    model.add(tf.keras.layers.Dense(128, activation='sigmoid'))
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))


    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def create_pretext_model_dl(self, input_shape, num_classes):
    
    model = tf.keras.models.Sequential()
    # model.add(tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01), activation='linear', input_dim=input_shape))
    model.add(tf.keras.layers.Dense(1000, activation='linear', input_dim=input_shape))
    # model.add(tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.l1_l2(0.01), activation='sigmoid', input_dim=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    # model.add(tf.keras.layers.Dense(128, activation='sigmoid', input_dim=input_shape))
    # model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(input_shape, activation='relu'))

    opt = tf.keras.optimizers.RMSprop()

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    self.latent_layer = 1

    return model
  
  def get_latent_space(self, data):

    get_3rd_layer_output = K.function([self.final_model.layers[0].input],
                                      [self.final_model.layers[self.latent_layer].output])

    layer_outs = get_3rd_layer_output([data])
    return np.array(layer_outs[0])

  
  # def predict(self, X):

  #   get_3rd_layer_output = K.function([self.final_pretext_model.layers[0].input],
  #                                     [self.final_pretext_model.layers[1].output])

  #   transformedX = np.array(X)
  #   layer_outs = get_3rd_layer_output([transformedX])
      
  #   y_pred = np.argmax(self.final_model.predict(layer_outs), axis=1)
  #   return y_pred

  def create(
    self,
    n_rounds=0,
    name='centralized',
    epochs_client=1,
    epochs_server=300,
    verbose_client=0,
    verbose_server=1):
    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      clients_dataX = self.ssl_fl_problem.clients_dataX
      clients_dataY = self.ssl_fl_problem.clients_dataY
      input_shape = self.ssl_fl_problem.input_shape
      num_classes = self.ssl_fl_problem.num_classes

      n_clients = len(clients_dataX)
      model_list = [self.create_pretext_model_dl(input_shape, num_classes) for i in range(n_clients)]
      final_pretext_model = self.create_pretext_model_dl(input_shape, num_classes)
      final_model = self.create_model_dl(input_shape, num_classes)
      # final_model = self.create_model_dl(1000, num_classes)
      
      # n_rounds = 50
      # n_rounds = 10 # BRACIS Experiment
      # n_rounds = 3
      # n_rounds = 1
      # self.name = 'fssl'
      # self.name = 'fssl-noniid'

      self.name = name
      if name == 'centralized':
        n_rounds = 0
      
      log_autoencoder_weights_init_rounds = []
      log_autoencoder_weights_final_rounds = []
      log_final_pretext_model_weights_init_rounds = []
      log_final_pretext_model_weights_final_rounds = []
      
      log_rounds_autoencoder = []
      log_rounds_final_model = [] # TODO: Add option to test final model at each round

      log_final_model = None
      
      # clients_per_round = n_clients
      # clients_per_round = 5
      clients_per_round = 1
      eval_reconstruct_train_list = []
      eval_reconstruct_test_list = []
      for i in range(n_rounds):
        log_round_autoencoder = []
        print("Round ", i)
        round_clients_id = np.arange(len(clients_dataX))
        np.random.shuffle(round_clients_id)

        log_autoencoder_weights_init_rounds.append([])
        log_autoencoder_weights_final_rounds.append([])
        log_final_pretext_model_weights_init_rounds.append(final_pretext_model.get_weights())

        # for client_id, (m, clientX, clientY) in enumerate(zip(model_list, clients_dataX, clients_dataY)):
        for j, client_id in enumerate(round_clients_id[:clients_per_round]):

          print("Client {} (real client {})".format(j, client_id))
          m = model_list[client_id]
          clientX = clients_dataX[client_id]
          clientY = clients_dataY[client_id]


          m.set_weights(final_pretext_model.get_weights())
          
          # m.fit(clientX, clientX, verbose=0)
          log_autoencoder_weights_init_rounds[-1].append(m.get_weights())
          log = m.fit(clientX, clientX, epochs=epochs_client, verbose=verbose_client)
          log_autoencoder_weights_final_rounds[-1].append(m.get_weights())

          log_round_autoencoder.append(log.history)
        
        new_weights = []
        for lidx in range(len(final_pretext_model.get_weights())):
          # avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          avg_weights = sum([np.array(model_list[x].get_weights()[lidx]) for x in round_clients_id[:clients_per_round]])/clients_per_round
          new_weights.append(avg_weights)

        final_pretext_model.set_weights(new_weights)
        
        log_final_pretext_model_weights_final_rounds.append(final_pretext_model.get_weights())
        log_rounds_autoencoder.append(log_round_autoencoder)
        
        # gs = log_final_pretext_model_weights_init_rounds[-1][0] - log_final_pretext_model_weights_final_rounds[-1][0]
        # gc = log_autoencoder_weights_init_rounds[-1][0][0] - log_autoencoder_weights_final_rounds[-1][0][0]
        # print(np.sum(np.abs(gs-gc)))
        # exit()
        

        eval_reconstruct_train = final_pretext_model.evaluate(self.ssl_fl_problem.trainX, self.ssl_fl_problem.trainX)
        eval_reconstruct_test = final_pretext_model.evaluate(self.ssl_fl_problem.testX, self.ssl_fl_problem.testX)
        eval_reconstruct_train_list.append(eval_reconstruct_train)
        eval_reconstruct_test_list.append(eval_reconstruct_test)

      print(eval_reconstruct_train_list)
      print(eval_reconstruct_test_list)
      self.final_pretext_model = final_pretext_model
      
      # get_3rd_layer_output = K.function([self.final_pretext_model.layers[0].input],
      #                                 [self.final_pretext_model.layers[1].output])

      # transformed_trainX = np.array(self.ssl_fl_problem.trainX)
      # transformed_trainX = get_3rd_layer_output([transformed_trainX])


      new_weights = final_model.get_weights()
      encoder_layer_limit = 2
      for lidx in range(len(final_pretext_model.get_weights())):
        if lidx == encoder_layer_limit:
          break
        
        avg_weights = final_pretext_model.get_weights()[lidx]
        new_weights[lidx] = avg_weights

      final_model.set_weights(new_weights)
      
      categY = tf.keras.utils.to_categorical(self.ssl_fl_problem.trainY, num_classes = num_classes)

      # log_final_model = final_model.fit(transformed_trainX, categY, epochs=epochs_server, verbose=verbose_server)
      log_final_model = final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=epochs_server, verbose=verbose_server)
      
      # print(final_model.layers[0].input)
      
      self.final_model = final_model
      log_final_model_weights = self.final_model.get_weights()

      return {
        'final_model_weights': log_final_model_weights,
        'log_autoencoder_weights_init_rounds': log_autoencoder_weights_init_rounds,
        'log_autoencoder_weights_final_rounds': log_autoencoder_weights_final_rounds,
        'log_final_pretext_model_weights_init_rounds': log_final_pretext_model_weights_init_rounds,
        'log_final_pretext_model_weights_final_rounds': log_final_pretext_model_weights_final_rounds,
        'log_rounds_autoencoder': log_rounds_autoencoder,
        'log_rounds_final_model': log_rounds_final_model,
        'log_final_model': log_final_model.history,
        'eval_reconstruct_train': eval_reconstruct_train_list,
        'eval_reconstruct_test': eval_reconstruct_test_list,
      }


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()

if __name__ == "__main__":
  main()