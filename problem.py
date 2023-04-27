import pandas as pd
import numpy as np
import argparse


from cybersecurity_datasets import load_nslkdd, load_toniot, load_botiot
# from cybersecurity_datasets import load_toniot, load_botiot
# from cybersecurity_datasets import load_toniot

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# import gc
# from sklearn import preprocessing

# import matplotlib
# import matplotlib.pyplot as plt

# from os.path import isfile, join
# import pickle


# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# from sklearn.manifold import TSNE
# # from tsnecuda import TSNE
# from sklearn.decomposition import PCA
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import chi2
# from sklearn.feature_selection import SelectFromModel

# from sklearn.multiclass import OneVsRestClassifier
# from sklearn import svm, datasets
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.metrics import classification_report, confusion_matrix

# import time

# import json

# import networkx as nx
 
import tensorflow as tf
# # import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

# from sklearn import preprocessing


# import warnings

# import seaborn as sn
# import pandas as pd

# import imblearn
# from sklearn.metrics import precision_recall_fscore_support

# from sklearn.decomposition import PCA

from fed_bench_utils import partition_data

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

class SSLFLProblem(object):
  def __init__(self, clients_dataX, trainX, trainY, testX, testY, clients_dataY=None):
    self.clients_dataX = clients_dataX
    self.clients_dataY = clients_dataY
    self.trainX = trainX
    self.trainY = trainY
    self.testX = testX
    self.testY = testY

    self.data_type = type(self.trainX[0][0])
    self.input_shape = self.trainX.shape[1]
    self.num_classes = len(set(self.trainY))

  def report_metrics(self, ssl_fl_solution):
    # s = ssl_fl_solution.final_model.score(self.testX, self.testY)
    # y_pred = ssl_fl_solution.final_model.predict(self.testX)
    y_pred = ssl_fl_solution.predict(self.testX)
    report = classification_report(self.testY, y_pred)

    print(report)

  def report_metrics_on_cross_val(self, solution, n_folds=10):
    # print("\n\SimpleSSLFLSolution")
    # s = Solution(self)
    # s.create()
    # s.report_metrics()

    total_dataX = np.concatenate((self.trainX, self.testX))
    total_dataY = np.concatenate((self.trainY, self.testY))

    skf = StratifiedKFold(n_splits=n_folds)

    old_trainX = self.trainX
    old_trainY = self.trainY
    old_testX = self.testX
    old_testY = self.testY
    
    # for i, (train_index, test_index) in enumerate(skf.split(total_dataX, total_dataY)):
    
    for i, (test_index, train_index) in enumerate(skf.split(total_dataX, total_dataY)): # test index is bigger (inverted leave one out)
      self.trainX = total_dataX[train_index]
      self.trainY = total_dataY[train_index]
      self.testX = total_dataX[test_index]
      self.testY = total_dataY[test_index]

      print(f"Fold {i}")
      
      solution.create()
      solution.report_metrics()

    self.trainX = old_trainX
    self.trainY = old_trainY
    self.testX = old_testX
    self.testY = old_testY

class RandomGeneratedProblem(SSLFLProblem):
  def __init__(self):
    n_clients = 10
    X, y = make_classification(n_samples=1200, n_features=10,
                               n_classes=5,
                                n_informative=4, n_redundant=0,
                                random_state=0, shuffle=False)
    
    clients_dataX = []
    clients_dataY = []

    skf = StratifiedKFold(n_splits=n_clients+2)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
      # # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)
      # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=0)
      # X_over, y_over = undersample.fit_resample(total_data_X[test_index], total_data_Y[test_index])
      
      # fl_dataX[f'Client {i}'] = X_over
      # fl_dataY[f'Client {i}'] = y_over

      clients_dataX.append(X[test_index])
      clients_dataY.append(y[test_index])
    
    

    # trainX, testX, trainY, testY = train_test_split(
    #   X, y, test_size=0.33, random_state=42)
    trainX = clients_dataX[0]
    trainY = clients_dataY[0]
    
    testX = clients_dataX[1]
    testY = clients_dataY[1]
    
    clients_dataX = clients_dataX[2:]
    clients_dataY = clients_dataY[2:]
    

    

    SSLFLProblem.__init__(self, clients_dataX, trainX, trainY, testX, testY, clients_dataY=clients_dataY)

class NSLKDDProblem(SSLFLProblem):
  def __init__(self, input_file):
    arg_test = [
      {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.01 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
      # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.8 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    ]
    
    # n_clients = 50
    n_clients = 10
    random_seed = 0

    data_set_name = 'toniot'
    args = arg_test[0]

    ton_args = arg_test[0]
    
    drop_cols = ton_args['drop_cols']
    class_type = ton_args['class_type']
    test_ratio = ton_args['test_ratio']
    return_fnames = ton_args['return_fnames']
    use_dimred = ton_args['use_dimred']

    class_red_type = 'no_fselect'
    if 'class_red_type' in ton_args:
      class_red_type = ton_args['class_red_type']

    
    if len(drop_cols) > 0:
      drop_col_names = 'drop_' + '_'.join(drop_cols)
    else:
      drop_col_names = ''
    
    ret_fnames_name = 'return_fnames' if return_fnames else ''
    use_dimred_name = f'use_dimred_{class_red_type}' if use_dimred else ''

    # if class_type != 'one_attack':
    
    # if not ('attack_type' in ton_args):
    #   ds_name = f'toniot_{use_dimred_name}_{class_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
    # else:
    #   attack_type = ton_args['attack_type']
    #   ds_name = f'toniot_{use_dimred_name}_one_attack_{attack_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'



    ton_args_copy = ton_args.copy()

    if 'use_dimred' in ton_args_copy:
      del ton_args_copy['use_dimred']
    if 'class_red_type' in ton_args_copy:
      del ton_args_copy['class_red_type']

    if data_set_name == 'toniot':
      trainX, trainY, testX, testY, feature_names, ds_name = load_toniot(input_file, **ton_args_copy)
    
    if data_set_name == 'botiot':
      trainX, trainY, testX, testY, feature_names, ds_name = load_botiot(
        args.input_file, **ton_args_copy)

    if data_set_name == 'nsl-kdd':
      trainX, trainY, testX, testY, feature_names, ds_name = load_nslkdd()

    data_augmentation = None
    # dirichlet_beta = 100
    dirichlet_beta = 0.1

    src_ip_idx = None
    dst_ip_idx = None
    if data_set_name == 'toniot':
      src_ip_idx = feature_names.index('src_ip')
      dst_ip_idx = feature_names.index('dst_ip')
    if data_set_name == 'botiot':
      src_ip_idx = feature_names.index('saddr')
      dst_ip_idx = feature_names.index('daddr')

    if (not src_ip_idx is None) and (not dst_ip_idx is None):
      ip_set = set(trainX[:, src_ip_idx]).union(set(trainX[:, dst_ip_idx]))
      ip_set = list(ip_set)



    if (not src_ip_idx is None) and (not dst_ip_idx is None):
      trainX = np.delete(trainX, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      testX = np.delete(testX, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
    else:
      trainX = trainX.astype(np.float32)
      testX = testX.astype(np.float32)

    # scaler = MinMaxScaler()
    # scaler.fit(np.concatenate((trainX, testX)))
    # trainX = scaler.transform(trainX)
    # testX = scaler.transform(testX)


    # scaler = Normalizer()
    # scaler.fit(np.concatenate((trainX, testX)))
    # trainX = scaler.transform(trainX)
    # testX = scaler.transform(testX)


    le = LabelEncoder()
    le.fit(np.concatenate((trainY, testY)))
    trainY = le.transform(trainY)
    testY = le.transform(testY)

    # TODO: Remove str features


    # client_dataX, server_dataX, client_dataY, server_dataY = train_test_split(
    # trainX, trainY, test_size=0.1, random_state=42,
    # stratify=trainY
    #   # shuffle=False
    # )

    client_dataX, server_dataX, client_dataY, server_dataY = train_test_split(
    trainX, trainY, test_size=0.0009, random_state=42,
    stratify=trainY,
    shuffle=True
    )

    # print(len((client_dataX)))
    # print(len((server_dataX)))
    # print(len((testX)))
    # exit()

    d = partition_data(
      (trainX, trainY, trainX, trainY),
      "",
      "logdir/",
      "noniid-labeldir", # iid-diff-quantity, noniid-labeldir
      # 5
      # 16,
      # 100,
      n_clients,
      beta=dirichlet_beta,
      random_seed=random_seed
    )

    mask_total_data_X_local_ips = np.ones(trainX.shape[0], dtype=np.bool)

    
    # fl_dataX = {i: trainX[d[4][i]] for i in d[4]}
    # fl_dataY = {i: trainY[d[4][i]] for i in d[4]}
    fl_dataX = [trainX[d[4][i]] for i in d[4]]
    fl_dataY = [trainY[d[4][i]] for i in d[4]]

    # print(len(fl_dataX))
    # print([len(fl_dataX[x]) for x in fl_dataX])

    if data_augmentation == 'oversampling':
      for i in fl_dataX:
        if len(set(fl_dataY[i])) > 1:
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='all', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)

          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='majority', random_state=0)
          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='not minority', random_state=0)
          undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='all', random_state=0)
          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='not majority', random_state=0)

          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            x_ips = fl_dataX[i][:, [src_ip_idx, dst_ip_idx]]
            x_feat = np.delete(fl_dataX[i], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
          
          print("Initiating undersample with {} samples".format(len(x_feat)))
          # undersample = imblearn.under_sampling.CondensedNearestNeighbour(random_state=0, sampling_strategy='not minority')
          # undersample = imblearn.under_sampling.EditedNearestNeighbours()
          # undersample = imblearn.under_sampling.RepeatedEditedNearestNeighbours()
          # undersample = imblearn.under_sampling.AllKNN()
          # undersample = imblearn.under_sampling.InstanceHardnessThreshold()
          # undersample = imblearn.under_sampling.NearMiss()
          # undersample = imblearn.under_sampling.NeighbourhoodCleaningRule()
          # undersample = imblearn.under_sampling.OneSidedSelection()
          # undersample = imblearn.under_sampling.TomekLinks()
          X_over, y_over = undersample.fit_resample(x_feat, fl_dataY[i])

          if len(x_ips) < len(X_over):
            new_x_ips = np.zeros((len(X_over), 2)).astype(str)
            new_x_ips[:len(x_ips), :] = x_ips[:, :]
            new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
            new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
            x_ips = new_x_ips
          
          # X_over = np.concatenate([x_ips[:len(X_over), :], X_over], axis=1) # TODO: Get real IP values
          
          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            if src_ip_idx < dst_ip_idx:
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
            else:
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

          # print(X_over)
          # exit()
          print("Ending undersample with {} samples".format(len(X_over)))
          
          fl_dataX[i] = X_over
          fl_dataY[i] = y_over

    if data_augmentation == 'test':
      for i in fl_dataX:
        if len(set(fl_dataY[i])) > 1:
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:100 for x in set(fl_dataY[i])}, random_state=0)
          undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:10 for x in set(fl_dataY[i])}, random_state=0)

          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            x_ips = fl_dataX[i][:, [src_ip_idx, dst_ip_idx]]
            x_feat = np.delete(fl_dataX[i], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
          
          print("Initiating undersample with {} samples".format(len(x_feat)))
          X_over, y_over = undersample.fit_resample(x_feat, fl_dataY[i])
          print(X_over.shape)

          if len(x_ips) < len(X_over):
            new_x_ips = np.zeros((len(X_over), 2)).astype(str)
            new_x_ips[:len(x_ips), :] = x_ips[:, :]
            new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
            new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
            x_ips = new_x_ips
          
          if (not src_ip_idx is None) and (not dst_ip_idx is None):

            if src_ip_idx < dst_ip_idx:
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
            else:
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

          print("Ending undersample with {} samples".format(len(X_over)))
          
          fl_dataX[i] = X_over
          fl_dataY[i] = y_over

            
    SSLFLProblem.__init__(self, fl_dataX, server_dataX, server_dataY, testX, testY, clients_dataY=fl_dataY)
    return

    # print(len(trainX))
    # exit()

    
    X, y = make_classification(n_samples=1200, n_features=10,
                               n_classes=5,
                                n_informative=4, n_redundant=0,
                                random_state=0, shuffle=False)
    
    clients_dataX = []
    clients_dataY = []

    skf = StratifiedKFold(n_splits=n_clients+2)
    
    for i, (train_index, test_index) in enumerate(skf.split(X, y)):
      # # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)
      # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=0)
      # X_over, y_over = undersample.fit_resample(total_data_X[test_index], total_data_Y[test_index])
      
      # fl_dataX[f'Client {i}'] = X_over
      # fl_dataY[f'Client {i}'] = y_over

      clients_dataX.append(X[test_index])
      clients_dataY.append(y[test_index])
    
    

    # trainX, testX, trainY, testY = train_test_split(
    #   X, y, test_size=0.33, random_state=42)
    trainX = clients_dataX[0]
    trainY = clients_dataY[0]
    
    testX = clients_dataX[1]
    testY = clients_dataY[1]
    
    clients_dataX = clients_dataX[2:]
    clients_dataY = clients_dataY[2:]
    

    

    SSLFLProblem.__init__(self, clients_dataX, trainX, trainY, testX, testY, clients_dataY=clients_dataY)

class SSLFLSolution(object):
  def __init__(self, ssl_fl_problem):
    self.ssl_fl_problem = ssl_fl_problem
    self.final_model = None

  def report_metrics(self):
    self.ssl_fl_problem.report_metrics(self)

  def predict(self, X):
    return self.final_model.predict(X)
    
  def report_metrics_on_cross_val(self, n_folds = 10):
    self.ssl_fl_problem.report_metrics_on_cross_val(self, n_folds=n_folds)

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
          # print(categY)
          # print(clientX.shape)
          m.fit(clientX, categY)
          
          # model_list.append(m)

        new_weights = []
        for lidx in range(len(final_model.get_weights())):
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          new_weights.append(avg_weights)

        # final_model.set_weights(avg_weights)
        final_model.set_weights(new_weights)

      self.final_model = final_model

  def predict(self, X):
    y_pred = np.argmax(self.final_model.predict(X), axis=1)
    return y_pred



class SimpleSSLFLSolution(SSLFLSolution):
  def __init__(self, ssl_fl_problem):
    SSLFLSolution.__init__(self, ssl_fl_problem)
  
  def create_model_dl(self, input_shape, num_classes):
    '''
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='relu', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    '''
    
    
    model = tf.keras.models.Sequential()
    
    # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape,
    # kernel_regularizer=tf.keras.regularizers.l2()))

    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))


    model.add(tf.keras.layers.Dropout(0.2))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.SGD(learning_rate=1)
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

  def create_pretext_model_dl(self, input_shape, num_classes):
    '''
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='relu', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(input_shape, activation='linear'))

    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    opt = tf.keras.optimizers.SGD(learning_rate=0.000001)
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    return model
    '''

    '''
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(input_shape, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    '''

    model = tf.keras.models.Sequential()
    
    
    model.add(tf.keras.layers.Dense(1000, activation='linear', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dropout(0.2))


    model.add(tf.keras.layers.Dense(input_shape, activation='relu'))

    # opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    # opt = tf.keras.optimizers.SGD(learning_rate=0.000001)
    opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

  def create(self):
    if self.ssl_fl_problem.data_type in [float, np.float32, np.float64]:
      clients_dataX = self.ssl_fl_problem.clients_dataX
      clients_dataY = self.ssl_fl_problem.clients_dataY
      input_shape = self.ssl_fl_problem.input_shape
      num_classes = self.ssl_fl_problem.num_classes

      n_clients = len(clients_dataX)
      # model_list = [self.create_model_dl(input_shape, num_classes) for i in range(n_clients)]
      model_list = [self.create_pretext_model_dl(input_shape, num_classes) for i in range(n_clients)]
      final_pretext_model = self.create_pretext_model_dl(input_shape, num_classes)
      final_model = self.create_model_dl(input_shape, num_classes)
      
      # n_rounds = 50
      n_rounds = 10
      # n_rounds = 3
      # n_rounds = 1
      # n_rounds = 0
      
      for i in range(n_rounds):
        print("Round ", i)
        for m, clientX, clientY in zip(model_list, clients_dataX, clients_dataY):
          m.set_weights(final_pretext_model.get_weights())
          
          # categY = tf.keras.utils.to_categorical(clientY, num_classes = num_classes)

          # m.fit(clientX, categY)
          m.fit(clientX, clientX, verbose=0)
          
          # model_list.append(m)

        # avg_weights = sum([np.array(x.get_weights()) for x in model_list])/len(model_list)
        # final_model.set_weights(avg_weights)
        
        new_weights = []
        for lidx in range(len(final_pretext_model.get_weights())):
          avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
          new_weights.append(avg_weights)

        # final_model.set_weights(avg_weights)
        final_pretext_model.set_weights(new_weights)
        
      new_weights = final_model.get_weights()
      # print([x.shape for x in final_model.get_weights()])
      encoder_layer_limit = 2
      for lidx in range(len(final_pretext_model.get_weights())):
        if lidx == encoder_layer_limit:
          break
        
        avg_weights = sum([np.array(x.get_weights()[lidx]) for x in model_list])/len(model_list)
        # new_weights.append(avg_weights)
        new_weights[lidx] = avg_weights

      # final_model.set_weights(avg_weights)
      final_model.set_weights(new_weights)

      categY = tf.keras.utils.to_categorical(self.ssl_fl_problem.trainY, num_classes = num_classes)

      # final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=1)
      # final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=100)
      final_model.fit(self.ssl_fl_problem.trainX, categY, epochs=300, verbose=0)
      # for i in range(100):
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


class SimpleNonFLSolution(SSLFLSolution):
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
  # parser.add_argument('-o', '--output_file', dest='output_file', type=str,
  #                     required=False, default='result_toniot.json')
  # parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
  #                     type=int)
  # parser.add_argument('-s', dest='silent', action='store_true')
  # parser.add_argument('-dl', '--use_dl_model', dest='use_dl_model', action='store_true')

  
  # parser.add_argument('-t', '--data_seg_type', dest='data_seg_type', type=str,
  #                     required=False, choices=['ip', 'iid', 'noniid', 'dirichlet'], default='ip')
  # parser.add_argument('-db', '--dirichlet_beta', dest='dirichlet_beta', type=float,
  #                     required=False, default=0.4)
  # parser.add_argument('-nc', '--n_clients', dest='n_clients', type=int,
  #                     required=False, default=16)
  # parser.add_argument('-rs', '--random_seed', dest='random_seed', type=int,
  #                     required=False, default=0)


  # parser.add_argument('-vt', '--vote_type', dest='vote_type', type=str,
  #                     required=False, choices=['hard', 'norm', 'weight_norm', 'relative_weight_norm', 'soft_rank'], default='hard')

  # parser.add_argument('-da', '--data_augmentation', dest='data_augmentation', type=str,
  #                     required=False, choices=['none', 'oversampling', 'test'], default='none')

  
  # parser.add_argument('-d', '--data_set_name', dest='data_set_name', type=str,
  #                     required=False, choices=['toniot', 'botiot', 'nsl-kdd'], default='toniot')

  # parser.set_defaults(list=[])    
  # parser.set_defaults(silent=False)
  # parser.set_defaults(use_dl_model=False)
  
  args = parser.parse_args()


  # p = RandomGeneratedProblem()
  # print("RandomGeneratedProblem")
  # print("#"*80)

  p = NSLKDDProblem(args.input_file)
  print("NSLKDDProblem")
  print("#"*80)


  # print("SimpleNonFLSolution")
  # s = SimpleNonFLSolution(p)
  # s.create()
  # s.report_metrics()

  # print("\n\SimpleFLSolution")
  # s = SimpleFLSolution(p)
  # s.create()
  # s.report_metrics()

  print("\n\SimpleSSLFLSolution")
  s = SimpleSSLFLSolution(p)
  # s.create()
  # s.report_metrics()
  s.report_metrics_on_cross_val()


  exit()

  
  print(args.input_file, args.list, args.silent, args.output_file)
  data_seg_type = args.data_seg_type
  dirichlet_beta = args.dirichlet_beta
  n_clients = args.n_clients
  vote_type = args.vote_type
  random_seed = args.random_seed
  data_set_name = args.data_set_name
  use_dl_model = args.use_dl_model
  data_augmentation = args.data_augmentation

  # print(n_clients)
  # exit()

  '''
  arg_test = [
    
    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},

    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts']},
    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    
    # TonIot
    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.444 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    
    # BotIot
    {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.444 , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},

    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'normal', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
     
  ]
  '''

  if data_set_name == 'toniot':
    arg_test = [
      {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.444 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    ]
  
  if data_set_name == 'botiot':
    arg_test = [
      {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.444 , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},
    ]
  
  if data_set_name == 'nsl-kdd':
    arg_test = [
      {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.444 , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},
    ]
  
  np.random.seed(random_seed)


  result_data = []

  if isfile(args.output_file):
    with open(args.output_file, 'r') as f:
      result_data = json.load(f)

  ton_args = arg_test[0]
  
  
  
  drop_cols = ton_args['drop_cols']
  class_type = ton_args['class_type']
  test_ratio = ton_args['test_ratio']
  return_fnames = ton_args['return_fnames']
  use_dimred = ton_args['use_dimred']

  class_red_type = 'no_fselect'
  if 'class_red_type' in ton_args:
    class_red_type = ton_args['class_red_type']

  
  if len(drop_cols) > 0:
    drop_col_names = 'drop_' + '_'.join(drop_cols)
  else:
    drop_col_names = ''
  
  ret_fnames_name = 'return_fnames' if return_fnames else ''
  use_dimred_name = f'use_dimred_{class_red_type}' if use_dimred else ''

  # if class_type != 'one_attack':
  
  # if not ('attack_type' in ton_args):
  #   ds_name = f'toniot_{use_dimred_name}_{class_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
  # else:
  #   attack_type = ton_args['attack_type']
  #   ds_name = f'toniot_{use_dimred_name}_one_attack_{attack_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'



  ton_args_copy = ton_args.copy()

  if 'use_dimred' in ton_args_copy:
    del ton_args_copy['use_dimred']
  if 'class_red_type' in ton_args_copy:
    del ton_args_copy['class_red_type']

  if data_set_name == 'toniot':
    trainX, trainY, testX, testY, feature_names, ds_name = load_toniot(args.input_file, **ton_args_copy)
  
  if data_set_name == 'botiot':
    trainX, trainY, testX, testY, feature_names, ds_name = load_botiot(
      args.input_file, **ton_args_copy)

  if data_set_name == 'nsl-kdd':
    trainX, trainY, testX, testY, feature_names, ds_name = load_nslkdd()

  print(ds_name)

  # continue

  # print(set(testY))
  # exit()
  
  # total_data_X = np.concatenate([trainX, testX])
  # total_data_Y = np.concatenate([trainY, testY])
  
  total_data_X = np.concatenate([trainX])
  total_data_Y = np.concatenate([trainY])
  
  total_data_X_test = np.concatenate([testX])
  total_data_Y_test = np.concatenate([testY])

  
  if data_set_name == 'toniot':
    selected_classes = ['normal', 'password', 'backdoor', 'injection', 'xss', 'scanning']
  if data_set_name == 'botiot':    
    selected_classes = ['Theft', 'Normal', 'DoS', 'Reconnaissance', 'DDoS']
  if data_set_name == 'nsl-kdd':    
    selected_classes = ['phf', 'ipsweep', 'satan', 'pod', 'smurf', 'neptune', 'teardrop', 'loadmodule', 'imap', 'ftp_write', 'buffer_overflow', 'multihop', 'normal', 'nmap', 'perl', 'spy', 'warezclient', 'land', 'portsweep', 'guess_passwd', 'warezmaster', 'back', 'rootkit']

  print(set(list(total_data_Y)))
  mask_sel = np.zeros(len(total_data_Y)).astype(np.bool)
  mask_sel_test = np.zeros(len(total_data_Y_test)).astype(np.bool)
  for c in selected_classes:
    mask_sel = np.logical_or(mask_sel, total_data_Y==c)
    mask_sel_test = np.logical_or(mask_sel_test, total_data_Y_test==c)

  print(total_data_X.shape)
  # exit()


  total_data_X = total_data_X[mask_sel].astype(np.float32)
  total_data_Y = total_data_Y[mask_sel]
  
  # print(list(total_data_Y[:100]))
  # exit()


  total_data_X_test = total_data_X_test[mask_sel_test].astype(np.float32)
  total_data_Y_test = total_data_Y_test[mask_sel_test]

  # print(type(total_data_X))
  # print(type(total_data_X_test))
  # exit()

  le = preprocessing.LabelEncoder()
  # le.fit(total_data_Y)
  le.fit(np.concatenate([total_data_Y, total_data_Y_test]))


  set_y = set(total_data_Y)
  # print(set_y)
  # exit()

  src_ip_idx = None
  dst_ip_idx = None
  if data_set_name == 'toniot':
    src_ip_idx = feature_names.index('src_ip')
    dst_ip_idx = feature_names.index('dst_ip')
  if data_set_name == 'botiot':
    src_ip_idx = feature_names.index('saddr')
    dst_ip_idx = feature_names.index('daddr')
        


  # print(src_ip_idx)
  # print(dst_ip_idx)
  # print(total_data_X[0])

  if (not src_ip_idx is None) and (not dst_ip_idx is None):
    ip_set = set(total_data_X[:, src_ip_idx]).union(set(total_data_X[:, dst_ip_idx]))
    ip_set = list(ip_set)
    # print(total_data_X)
    # exit()

    total_ips = len(ip_set)

  fl_dataX = {}
  fl_dataY = {}

  from fed_bench_utils import partition_data

  # unique, counts = np.unique(total_data_Y, return_counts=True)
  # count_dict = dict(zip(unique, counts))
  # print(count_dict)

  # '''
  if data_seg_type == 'dirichlet':
    d = partition_data(
      (total_data_X, total_data_Y, total_data_X, total_data_Y),
      "",
      "logdir/",
      "noniid-labeldir", # iid-diff-quantity, noniid-labeldir
      # 5
      # 16,
      # 100,
      n_clients,
      beta=dirichlet_beta,
      random_seed=random_seed
    )

    mask_total_data_X_local_ips = np.ones(total_data_X.shape[0], dtype=np.bool)

    fl_dataX = {i: total_data_X[d[4][i]] for i in d[4]}
    fl_dataY = {i: total_data_Y[d[4][i]] for i in d[4]}

    if data_augmentation == 'oversampling':
      for i in fl_dataX:
        if len(set(fl_dataY[i])) > 1:
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='all', random_state=0)
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)

          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='majority', random_state=0)
          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='not minority', random_state=0)
          undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='all', random_state=0)
          # undersample = imblearn.over_sampling.RandomOverSampler(sampling_strategy='not majority', random_state=0)

          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            x_ips = fl_dataX[i][:, [src_ip_idx, dst_ip_idx]]
            x_feat = np.delete(fl_dataX[i], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
          
          print("Initiating undersample with {} samples".format(len(x_feat)))
          # undersample = imblearn.under_sampling.CondensedNearestNeighbour(random_state=0, sampling_strategy='not minority')
          # undersample = imblearn.under_sampling.EditedNearestNeighbours()
          # undersample = imblearn.under_sampling.RepeatedEditedNearestNeighbours()
          # undersample = imblearn.under_sampling.AllKNN()
          # undersample = imblearn.under_sampling.InstanceHardnessThreshold()
          # undersample = imblearn.under_sampling.NearMiss()
          # undersample = imblearn.under_sampling.NeighbourhoodCleaningRule()
          # undersample = imblearn.under_sampling.OneSidedSelection()
          # undersample = imblearn.under_sampling.TomekLinks()
          X_over, y_over = undersample.fit_resample(x_feat, fl_dataY[i])

          if len(x_ips) < len(X_over):
            new_x_ips = np.zeros((len(X_over), 2)).astype(str)
            new_x_ips[:len(x_ips), :] = x_ips[:, :]
            new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
            new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
            x_ips = new_x_ips
          
          # X_over = np.concatenate([x_ips[:len(X_over), :], X_over], axis=1) # TODO: Get real IP values
          
          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            if src_ip_idx < dst_ip_idx:
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
            else:
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

          # print(X_over)
          # exit()
          print("Ending undersample with {} samples".format(len(X_over)))
          
          fl_dataX[i] = X_over
          fl_dataY[i] = y_over

    if data_augmentation == 'test':
      for i in fl_dataX:
        if len(set(fl_dataY[i])) > 1:
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:100 for x in set(fl_dataY[i])}, random_state=0)
          undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:10 for x in set(fl_dataY[i])}, random_state=0)

          if (not src_ip_idx is None) and (not dst_ip_idx is None):
            x_ips = fl_dataX[i][:, [src_ip_idx, dst_ip_idx]]
            x_feat = np.delete(fl_dataX[i], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
          
          print("Initiating undersample with {} samples".format(len(x_feat)))
          X_over, y_over = undersample.fit_resample(x_feat, fl_dataY[i])
          print(X_over.shape)

          if len(x_ips) < len(X_over):
            new_x_ips = np.zeros((len(X_over), 2)).astype(str)
            new_x_ips[:len(x_ips), :] = x_ips[:, :]
            new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
            new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
            x_ips = new_x_ips
          
          if (not src_ip_idx is None) and (not dst_ip_idx is None):

            if src_ip_idx < dst_ip_idx:
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
            else:
              X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
              X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

          print("Ending undersample with {} samples".format(len(X_over)))
          
          fl_dataX[i] = X_over
          fl_dataY[i] = y_over

            
  if data_seg_type == 'ip':
    # '''
    mask_total_data_X_local_ips = np.zeros(total_data_X.shape[0], dtype=np.bool)
    for ip in ip_set:
      if is_ip_private(ip):
        mask = total_data_X[:, src_ip_idx] == ip
        mask = np.logical_or(mask, total_data_X[:, dst_ip_idx] == ip)

        mask_total_data_X_local_ips = np.logical_or(mask_total_data_X_local_ips, mask)

        dataX = total_data_X[mask]

        # for i in range(dataX.shape[1]):
        #   if not i in [src_ip_idx, dst_ip_idx]:
        #     is_fin = np.isfinite(dataX[:, i].astype(np.float32))
        #     notis_fin = np.logical_not(is_fin)
        #     print(np.sum(notis_fin))

        dataY = total_data_Y[mask]
        fl_dataX[ip] = dataX
        fl_dataY[ip] = dataY
        unique, counts = np.unique(dataY, return_counts=True)
        print(dict(zip(unique, counts)))
    
    fl_dataX['internet'] = []
    fl_dataY['internet'] = []
    for x,y in zip(total_data_X, total_data_Y):
      if(not is_ip_private(x[src_ip_idx]) or (not is_ip_private(x[dst_ip_idx]))):
        fl_dataX['internet'].append(x)
        fl_dataY['internet'].append(y)

        
    
    fl_dataX['internet'] = np.array(fl_dataX['internet'])
    fl_dataY['internet'] = np.array(fl_dataY['internet'])
    unique, counts = np.unique(fl_dataY['internet'], return_counts=True)
    print(dict(zip(unique, counts)))
    # '''

    # if False and data_augmentation == 'test':
    if data_augmentation == 'test':
      for i in fl_dataX:
        if len(set(fl_dataY[i])) > 1:
          # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:100 for x in set(fl_dataY[i])}, random_state=0)
          undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:min(10, sum(fl_dataY[i]==x)) for x in set(fl_dataY[i])}, random_state=0)

          x_ips = fl_dataX[i][:, [src_ip_idx, dst_ip_idx]]
          x_feat = np.delete(fl_dataX[i], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
          
          print("Initiating undersample with {} samples".format(len(x_feat)))
          X_over, y_over = undersample.fit_resample(x_feat, fl_dataY[i])
          print(X_over.shape)

          if len(x_ips) < len(X_over):
            new_x_ips = np.zeros((len(X_over), 2)).astype(str)
            new_x_ips[:len(x_ips), :] = x_ips[:, :]
            new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
            new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
            x_ips = new_x_ips
          
          if src_ip_idx < dst_ip_idx:
            X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
            X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
          else:
            X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
            X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

          print("Ending undersample with {} samples".format(len(X_over)))
          
          fl_dataX[i] = X_over
          fl_dataY[i] = y_over

  if data_seg_type == 'noniid':

    # ''' Non IID
    mask_total_data_X_local_ips = np.ones(total_data_X.shape[0], dtype=np.bool)

    # total_clients = 1
    # total_clients = 2
    # total_clients = 4
    # total_clients = 8
    # total_clients = 16
    # total_clients = 32
    # total_clients = 64
    # total_clients = 10
    
    total_clients = n_clients

    p = np.random.permutation(len(total_data_X))
    total_data_X = total_data_X[p]
    total_data_Y = total_data_Y[p]

    chunksX = np.array_split(total_data_X, total_clients)
    chunksY = np.array_split(total_data_Y, total_clients)
    for i in range(total_clients):
      fl_dataX[f'Client {i}'] = chunksX[i]
      fl_dataY[f'Client {i}'] = chunksY[i]
      unique, counts = np.unique(chunksY[i], return_counts=True)
      print(dict(zip(unique, counts)))


    # from sklearn.ensemble import RandomForestClassifier
    # from sklearn.datasets import make_classification
    # X,y = fl_dataX[f'Client 0'], fl_dataY[f'Client 0']
    # X = np.delete(X, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
    # clf = RandomForestClassifier(random_state=0)
    # print(X.shape)
    # print(y.shape)
    # clf.fit(X, y)
    # print(clf.score(X,y))
    # exit()

    # print(fl_dataY)
    # exit()
    # '''

  if data_seg_type == 'iid':
    # ''' IID
    mask_total_data_X_local_ips = np.ones(total_data_X.shape[0], dtype=np.bool)

    # skf = StratifiedKFold(n_splits=2)
    # skf = StratifiedKFold(n_splits=4)
    # skf = StratifiedKFold(n_splits=8)
    # skf = StratifiedKFold(n_splits=16)
    # skf = StratifiedKFold(n_splits=32)
    # skf = StratifiedKFold(n_splits=64)
    
    skf = StratifiedKFold(n_splits=n_clients)
    
    for i, (train_index, test_index) in enumerate(skf.split(total_data_X, total_data_Y)):
      # # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='not minority', random_state=0)
      # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy='majority', random_state=0)
      # X_over, y_over = undersample.fit_resample(total_data_X[test_index], total_data_Y[test_index])
      
      # fl_dataX[f'Client {i}'] = X_over
      # fl_dataY[f'Client {i}'] = y_over

      fl_dataX[f'Client {i}'] = total_data_X[test_index]
      fl_dataY[f'Client {i}'] = total_data_Y[test_index]



    # '''

  # '''

  if data_augmentation == 'test':
    # undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:100 for x in set(total_data_Y_test)}, random_state=0)
    undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy={x:min(10, sum(total_data_Y_test==x)) for x in set(total_data_Y_test)}, random_state=0)

    x_ips = total_data_X_test[:, [src_ip_idx, dst_ip_idx]]
    x_feat = np.delete(total_data_X_test, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
    
    print("Initiating undersample with {} samples".format(len(x_feat)))
    X_over, y_over = undersample.fit_resample(x_feat, total_data_Y_test)
    print(X_over.shape)

    if len(x_ips) < len(X_over):
      new_x_ips = np.zeros((len(X_over), 2)).astype(str)
      new_x_ips[:len(x_ips), :] = x_ips[:, :]
      new_x_ips[len(x_ips):, 0] = np.random.choice(x_ips[:,0], len(X_over) - len(x_ips))
      new_x_ips[len(x_ips):, 1] = np.random.choice(x_ips[:,1], len(X_over) - len(x_ips))
      x_ips = new_x_ips
    
    if src_ip_idx < dst_ip_idx:
      X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))
      X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
    else:
      X_over = np.hstack((X_over[:,:dst_ip_idx], x_ips[:len(X_over), 1].reshape(-1,1), X_over[:,dst_ip_idx:]))
      X_over = np.hstack((X_over[:,:src_ip_idx], x_ips[:len(X_over), 0].reshape(-1,1), X_over[:,src_ip_idx:]))

    print("Ending undersample with {} samples".format(len(X_over)))
    
    total_data_X_test = X_over
    total_data_Y_test = y_over
    
  def create_model_dl(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    
    model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
  # '''

  # '''
  def create_model(input_shape, num_classes, clf_name='rdf'):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
    from sklearn.naive_bayes import BernoulliNB, MultinomialNB
    from sklearn.neural_network import MLPClassifier
    # X, y = make_classification(n_samples=1000, n_features=4,
    #                           n_informative=2, n_redundant=0,
    #                           random_state=0, shuffle=False)
    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    if clf_name == 'rdf':
      clf = RandomForestClassifier(max_depth=6)
    if clf_name == 'lgr':
      clf = LogisticRegression()
    if clf_name == 'gdb':
      clf = GradientBoostingClassifier()
    if clf_name == 'sgd':
      clf = SGDClassifier()
    if clf_name == 'bnb':
      clf = BernoulliNB()
    if clf_name == 'mnb':
      clf = MultinomialNB()
    if clf_name == 'mlp':
      # clf = MLPClassifier(verbose=1)
      clf = MLPClassifier(hidden_layer_sizes=(1000,))
    
    # clf = GradientBoostingClassifier()
    # clf.fit(X, y)

    # print(clf.predict([[0, 0, 0, 0]]))
    return clf
  # '''

  


  # num_classes = 10
  num_classes = len(set_y)
  # print(num_classes)
  # exit()

  # use_dl_model = False
  # use_dl_model = True
  
  clf_name = 'rdf'
  # clf_name = 'mlp'
  use_partial_fit = False

  if use_dl_model:
    use_partial_fit = False
  

  # print(fl_dataX)
  if (not src_ip_idx is None) and (not dst_ip_idx is None):
    input_shape = fl_dataX[list(fl_dataX)[0]].shape[1] - 2
  else:
    input_shape = fl_dataX[list(fl_dataX)[0]].shape[1]

  # latent_dim = int(np.sqrt(input_shape))
  # print(fl_dataX[0])
  # exit()

  # latent_dim = 1024
  # latent_dim = 256
  # latent_dim = 32
  latent_dim = 16
  # latent_dim = 2


  # rep_method = None
  # rep_method = 'avg_autoencoders'
  rep_method = 'aggregate_autoencoders'
  # rep_method = 'pca'
  
  if use_dl_model:
    # models = {x: create_model_dl(input_shape, num_classes) for x in fl_dataX}
    # avg_model = create_model_dl(input_shape, num_classes)
    if rep_method == 'aggregate_autoencoders':
      # models = {x: create_model_represetation((input_shape,), int(np.log(len(fl_dataX[x])))) for x in fl_dataX}
      models = {x: create_model_represetation((input_shape,), latent_dim) for x in fl_dataX}
    elif rep_method == 'pca':
      models = {x: PCA(n_components='mle') for x in fl_dataX}
    else:
      models = {x: create_model_represetation((input_shape,), latent_dim) for x in fl_dataX}

    avg_model = create_model_represetation((input_shape,), latent_dim)
  else:
    models = {x: create_model(input_shape, num_classes, clf_name=clf_name) for x in fl_dataX}
    avg_model = create_model(input_shape, num_classes, clf_name=clf_name)
  

  if use_partial_fit:
    avg_model.fit(np.random.random((num_classes, input_shape)), np.arange(num_classes))
    for x in models:
      models[x].fit(np.random.random((num_classes, input_shape)), np.arange(num_classes))
    

  total_rounds = 1

  if use_dl_model:
    # total_rounds = 50
    total_rounds = 10
    # total_rounds = 2
    # total_rounds = 1

  if rep_method == 'aggregate_autoencoders' or rep_method == 'pca':
    total_rounds = 1
  # for round_id in range(50):
  for round_id in range(total_rounds):
    print(f"Initiating round {round_id}")
    for x in models:
      if use_dl_model and rep_method == 'avg_autoencoders':
        models[x].set_weights(avg_model.get_weights())
      if not use_dl_model and use_partial_fit:
        # models[x].coefs_ = avg_model.coefs_
        pass
      if len(set(fl_dataY[x])) <= 1:
        continue
      
      model = models[x]
      dataX = fl_dataX[x]
      # np.random.shuffle(dataX)

      if (not src_ip_idx is None) and (not dst_ip_idx is None):
        dataX = np.delete(dataX, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      
      # dataX = np.delete(dataX, dst_ip_idx,axis=1).astype(np.float32)
      # print(set(list(dataY)))

      # dataY = fl_dataY[x]
      dataY = le.transform(fl_dataY[x]).astype(np.int32)
      if use_dl_model:
        # print(set(dataY))
        # print(dataY)
        # dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)
        new_dataY = np.zeros((len(dataY), num_classes))
        for y in range(num_classes):
          new_dataY[dataY==y, y] = 1
        dataY = new_dataY
        
        # model.fit(dataX, dataY, epochs=1, batch_size=1000, verbose=1, shuffle=True)
        # model.fit(dataX, dataX, epochs=10, batch_size=1000, verbose=1, shuffle=True)
        # model.fit(dataX, dataX, epochs=10, batch_size=32, verbose=1, shuffle=True
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        # verbose = 0
        verbose = 1

        if rep_method == 'avg_autoencoders':
          # model.fit(dataX, dataX, epochs=1000, batch_size=1000, verbose=verbose, shuffle=True, callbacks=[callback])
          model.fit(dataX, dataX, epochs=1, batch_size=1000, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=100, batch_size=1000, verbose=verbose, shuffle=True, callbacks=[callback])
        if rep_method == 'aggregate_autoencoders':
          # model.fit(dataX, dataX, epochs=0, batch_size=128, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=1, batch_size=128, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=3, batch_size=128, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=100, batch_size=128, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=1, batch_size=1, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=1, batch_size=8, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=100, batch_size=32, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=1000, batch_size=1024, verbose=verbose, shuffle=True, callbacks=[callback])
          # model.fit(dataX, dataX, epochs=min(1000, batch_size=1024, verbose=verbose, shuffle=True, callbacks=[callback]))

          dataY = np.array((fl_dataY[x]))
          sy = set(dataY)
          for i in range(50):
            for c in sy:
              m = dataY != c
              idx = np.where(dataY==c)[0]
              if len(idx) > 100:
                s = np.random.choice(idx, 100, replace=True)
              else:
                s = idx
              # print(sy, c, np.where(dataY==c))
              # print(dataY)
              m[list(s)] = True
              dataX = dataX[m]
              dataY = dataY[m]
            model.fit(dataX, dataX, epochs=10, batch_size=1024, verbose=verbose, shuffle=True, callbacks=[callback])
        
        if rep_method == 'pca':
          model.fit(dataX)

      else:
        if use_partial_fit:
          model.partial_fit(dataX, dataY)
          print(model.score(dataX, dataY))
        else:
          # print(dataY)
          model.fit(dataX, dataY)

          
      # to_drop = np.random.choice(len(other_id), max(0,len(other_id) - 100), replace=False)
      # to_drop = other_id[to_drop]
      
      # dataX = np.delete(dataX, to_drop, 0)
      # dataY = np.delete(dataY, to_drop, 0)

      # print(x)
      # model.fit(dataX, dataY, epochs=1, batch_size=1)
      # model.fit(dataX, dataY, epochs=1, batch_size=32)
      # model.fit(dataX, dataY, epochs=10, batch_size=64, verbose=1, shuffle=True)
      # print(len(dataY), len(set(list(fl_dataY[x]))))
      
      # model.fit(dataX, dataY)

    # '''
    # if use_dl_model:
    if use_dl_model and rep_method == 'avg_autoencoders':
      total_train_samples = sum([len(fl_dataX[x]) for x in fl_dataX])
      avg_contribution = {x: float(len(fl_dataX[x]))/total_train_samples for x in fl_dataX}
      
      avg_weights = sum([np.array(models[x].get_weights()) for x in models])/len(models)
      # avg_weights = sum([np.array(models[x].get_weights())*avg_contribution[x] for x in models])/len(models)
      avg_model.set_weights(avg_weights)
      print(np.sum([np.sum([np.sum(y.flatten()) for y in x]) for x in avg_model.get_weights()]))

      '''
      dataX = np.delete(total_data_X, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      dataY = le.transform(total_data_Y).astype(np.int32)
      new_dataY = np.zeros((len(dataY), num_classes))
      for y in range(num_classes):
        new_dataY[dataY==y, y] = 1
      dataY = new_dataY
      # dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)

      dataX_test = np.delete(total_data_X_test, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      dataY_test = le.transform(total_data_Y_test).astype(np.int32)
      # dataY_test = tf.keras.utils.to_categorical(dataY_test, num_classes = num_classes)
      new_dataY = np.zeros((len(dataY_test), num_classes))
      for y in range(num_classes):
        new_dataY[dataY_test==y, y] = 1
      dataY_test = new_dataY

      dataX = dataX[mask_total_data_X_local_ips]
      dataY = dataY[mask_total_data_X_local_ips]


      y_pred = np.argmax(avg_model.predict(dataX_test), axis=1)
      y_pred = le.inverse_transform(y_pred)
      
      dataY_test_nonsoft = np.argmax(dataY_test, axis=1)
      dataY_test_nonsoft = le.inverse_transform(dataY_test_nonsoft)

      acc = sum(np.array(y_pred) == dataY_test_nonsoft)/len(dataY_test)

      # cm = confusion_matrix(dataY_test_nonsoft, y_pred, labels=le.classes_)
      # print(classification_report(dataY_test_nonsoft, y_pred, target_names=le.classes_))
      final_y_true = dataY_test_nonsoft
      final_y_pred = y_pred

      print_method_result_metrics(final_y_true, final_y_pred, preamble=f'Round {round_id}')
      '''
      


    # '''
  
    
    

  if (not src_ip_idx is None) and (not dst_ip_idx is None):
    dataX = np.delete(total_data_X, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
  else:
    dataX = total_data_X.astype(np.float32)
  
  dataY = le.transform(total_data_Y).astype(np.int32)
  if use_dl_model:
    new_dataY = np.zeros((len(dataY), num_classes))
    for y in range(num_classes):
      new_dataY[dataY==y, y] = 1
    dataY = new_dataY
    # dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)
  else:
    dataY = total_data_Y

  if (not src_ip_idx is None) and (not dst_ip_idx is None):
    dataX_test = np.delete(total_data_X_test, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
  else:
    dataX_test = total_data_X_test.astype(np.float32)
        
  dataY_test = le.transform(total_data_Y_test).astype(np.int32)
  if use_dl_model:
    # dataY_test = tf.keras.utils.to_categorical(dataY_test, num_classes = num_classes)
    new_dataY = np.zeros((len(dataY_test), num_classes))
    for y in range(num_classes):
      new_dataY[dataY_test==y, y] = 1
    dataY_test = new_dataY
  else:
    dataY_test = total_data_Y_test
    # dataY_test = dataY_test

  # print(dataX.shape)
  # print(mask_total_data_X_local_ips.shape)
  dataX = dataX[mask_total_data_X_local_ips]
  dataY = dataY[mask_total_data_X_local_ips]
  
  print(np.sum([np.sum([np.sum(y.flatten()) for y in x]) for x in avg_model.get_weights()]))
  
  if not use_dl_model and not use_partial_fit:
  # '''
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    models_names = [x for x in models if len(set(fl_dataY[x])) > 1]
    # models_names = [x for x in models]
    # print(models_names)
    # exit()

    if vote_type == 'hard':
      avg_model = VotingClassifier(estimators=[(x, models[x]) for x in models_names], voting='hard')
      
      # for x in models_names:
      #   models[x].classes_ = le.classes_
      # avg_model = VotingClassifier(estimators=[(x, models[x]) for x in models_names], voting='soft')

      avg_model.estimators_ = [models[x] for x in models_names]
      # avg_model.le_ = LabelEncoder().fit(dataY)
      avg_model.le_ = le
      avg_model.classes_ = avg_model.le_.classes_
      acc = avg_model.score(dataX_test, dataY_test)
      
      y_pred = avg_model.predict(dataX_test)
      cm = confusion_matrix(dataY_test, y_pred, labels=le.classes_)
      print(classification_report(dataY_test, y_pred, target_names=le.classes_))
      final_y_true = dataY_test
      final_y_pred = y_pred


      # df_cm = pd.DataFrame(cm, index = le.classes_,
      #                      columns = le.classes_)
      # plt.figure(figsize = (10,7))
      # sn.heatmap(df_cm, annot=True)
      # plt.show()

    if vote_type == 'soft_rank':
      # avg_model = VotingClassifier(estimators=[(x, models[x]) for x in models_names], voting='hard')
      
      # for x in models_names:
      #   models[x].classes_ = le.classes_
      # avg_model = VotingClassifier(estimators=[(x, models[x]) for x in models_names], voting='soft')

      # avg_model.estimators_ = [models[x] for x in models_names]
      # avg_model.le_ = LabelEncoder().fit(dataY)
      # avg_model.le_ = le
      # avg_model.classes_ = avg_model.le_.classes_
      
      # y_pred = avg_model.predict(dataX_test)
      

      y_pred = []
      all_C = list(set_y)
      all_C_idx = {c: i for i, c in enumerate(all_C)}
      clf_preds = [models[x].predict_proba(dataX_test) for x in models_names]
      # clf_preds = [models[x].predict(dataX_test) for x in models_names]
      
      # import tqdm
      # for i in tqdm.tqdm(range(len(dataX_test))):
      for i in range(len(dataX_test)):
        votes = np.zeros(len(all_C))
        for m_idx, x in enumerate(models_names):
          probs = clf_preds[m_idx][i]
          argk = np.argsort(probs)[::-1]
          # argk = [probs]
          for j, c in enumerate(argk[:1]):
          # for j, c in enumerate(argk):
            # print(all_C_idx)
            # print(c)
            # print(models[x].classes_)
            # print(models[x].classes_[c])
            votes[all_C_idx[le.inverse_transform([models[x].classes_[c]])[0]]] += len(argk) - (j+1)
            # votes[all_C_idx[le.inverse_transform([models[x].classes_[c]])[0]]] += len(argk) - (j+1)
            # votes[all_C_idx[le.inverse_transform([models[x].classes_[c]])[0]]] += 1
            # votes[models[x].classes_[c]] += 1
            # votes[c] += 1
        
        # y_pred.append(le.inverse_transform([np.argmax(votes)])[0])

        y_pred.append(all_C[np.argmax(votes)])

      acc = sum(np.array(y_pred) == dataY_test)/len(dataY_test)
      # acc = avg_model.score(dataX_test, dataY_test)

      cm = confusion_matrix(dataY_test, y_pred, labels=le.classes_)
      print(classification_report(dataY_test, y_pred, target_names=le.classes_))
      final_y_true = dataY_test
      final_y_pred = y_pred



      # df_cm = pd.DataFrame(cm, index = le.classes_,
      #                      columns = le.classes_)
      # plt.figure(figsize = (10,7))
      # sn.heatmap(df_cm, annot=True)
      # plt.show()


    if vote_type == 'norm':
      # all_C = avg_model.le_.classes_
      all_C = list(set_y)
      C = [set(fl_dataY[x]) for x in models_names]
      TC = [len(x) for x in C]
      TA = [sum([c in Ci for Ci in C]) for c in all_C]
      # print(TA)
      # print(TC)
      # exit()

      clf_preds = [models[x].predict(dataX_test) for x in models_names]
      predictions = []
      for i in range(len(dataX_test)):
        votes = np.array(le.inverse_transform([x[i] for x in clf_preds]))
        TV = [sum(votes == c) for c in all_C]

        TP = [TV[i]/TA[i] for i in range(len(all_C))]
        predictions.append(all_C[np.argmax(TP)])

      acc = sum(np.array(predictions) == dataY_test)/len(dataY_test)

      cm = confusion_matrix(dataY_test, predictions, labels=le.classes_)
      print(classification_report(dataY_test, predictions, target_names=le.classes_))
      final_y_true = dataY_test
      final_y_pred = predictions

    if vote_type == 'weight_norm':
      all_C = list(set_y)
      C = [set(fl_dataY[x]) for x in models_names]
      TC = [len(x) for x in C]

      TCW = []
      TCW_norm = {}
      for x in models_names:
        TCW.append({})
        unique, counts = np.unique(fl_dataY[x], return_counts=True)
        count_dict = dict(zip(unique, counts))
        for c in all_C:
          if not c in TCW:
            TCW[-1][c] = 0

          if c in count_dict:
            TCW[-1][c] += count_dict[c]
            if c in TCW_norm:
              TCW_norm[c] += count_dict[c]
            else:
              TCW_norm[c] = count_dict[c]
          
        
      TA = [TCW_norm[c] if c in TCW_norm else np.inf for c in all_C] # Fix


      clf_preds = [models[x].predict(dataX_test) for x in models_names]
      predictions = []
      for i in range(len(dataX_test)):
        votes = np.array(le.inverse_transform([x[i] for x in clf_preds]))
        TV = np.zeros(len(all_C))
        for i, c in enumerate(all_C):
          for v, w in zip(votes, TCW):
            if v == c:
              TV[i] += w[c]
        TP = [TV[i]/TA[i] for i in range(len(all_C))]

        predictions.append(all_C[np.argmax(TP)])

      acc = sum(np.array(predictions) == dataY_test)/len(dataY_test)
      
      cm = confusion_matrix(dataY_test, predictions, labels=le.classes_)
      print(classification_report(dataY_test, predictions, target_names=le.classes_))
      final_y_true = dataY_test
      final_y_pred = predictions
      
    if vote_type == 'relative_weight_norm':
      all_C = list(set_y)
      C = [set(fl_dataY[x]) for x in models_names]
      TC = [len(x) for x in C]
      

      TCW = []
      TCW_norm = {}
      for x in models_names:
        TCW.append({})
        unique, counts = np.unique(fl_dataY[x], return_counts=True)
        count_dict = dict(zip(unique, counts))
        for c in all_C:
          if not c in TCW:
            TCW[-1][c] = 0

          if c in count_dict:
            TCW[-1][c] += count_dict[c]
            if c in TCW_norm:
              TCW_norm[c] += count_dict[c]
            else:
              TCW_norm[c] = count_dict[c]
          
        
      TA = [TCW_norm[c] if c in TCW_norm else np.inf for c in all_C] # Fix
      
      TA = np.array(TA)

      TA_norm = TA/np.sum(TA)



      clf_preds = [models[x].predict(dataX_test) for x in models_names]
      predictions = []
      for i in range(len(dataX_test)):
        votes = np.array(le.inverse_transform([x[i] for x in clf_preds]))
        TV = np.zeros(len(all_C))
        TV2 = np.zeros(len(all_C))
        for i, c in enumerate(all_C):
          for j, (v, w) in enumerate(zip(votes, TCW)):
            if v == c:
              TV2[i] += 1
            elif c in C[j]:
              TV[i] -= 1
        
        TP = [TV[i] for i in range(len(all_C))]

        predictions.append(all_C[np.argmax(TP)])

      acc = sum(np.array(predictions) == dataY_test)/len(dataY_test)
      cm = confusion_matrix(dataY_test, predictions, labels=le.classes_)
      print(classification_report(dataY_test, predictions, target_names=le.classes_))
      final_y_true = dataY_test
      final_y_pred = predictions
   
  if use_dl_model:
    # avg_weights = sum([np.array(models[x].get_weights()) for x in models])/len(models)

    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    
    '''
    train_dataX = np.delete(np.concatenate([fl_dataX[x] for x in fl_dataX]), np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
    train_dataY = np.concatenate([fl_dataY[x] for x in fl_dataY])
    
    transformed_dataX = avg_model.predict(dataX_test)
    transformed_train_dataX = avg_model.predict(train_dataX)

    clf = RandomForestClassifier()
    clf.fit(transformed_train_dataX, train_dataY)
    y_pred = np.array(clf.predict(transformed_dataX))
    # y_pred = le.inverse_transform(y_pred)
    
    '''
    dataY_test_nonsoft = np.argmax(dataY_test, axis=1)
    dataY_test_nonsoft = le.inverse_transform(dataY_test_nonsoft)
    

    # transformed_dataX = dataX_test.astype(np.float32)

    models_names = [x for x in models if len(set(fl_dataY[x])) > 1]
    
    models_sizes = sorted([len(fl_dataY[x]) for x in models_names])
    models_names = [x for x in models if len(fl_dataY[x]) <= models_sizes[2] ]
    
    models_fs = {x: None for x in models_names}

    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.pipeline import make_pipeline


      
    def get_concat_autoencoder(X, mn):
      # print(X.shape)
      # print(models[mn[0]].predict(X).shape)
      # print(models[mn[0]].get_latent_space(X).shape)
      # exit()

      # preds = [models[x].get_latent_space(X) for x in mn]
      # model_counts = [float(len(fl_dataY[x])) for x in mn]
      # model_counts = np.array(model_counts)
      
      # return np.array(models[mn[np.argsort(model_counts)[-1]]].get_latent_space(X))

      # preds = [X for x in mn]
      preds = [models[x].get_latent_space(X) for x in mn]
      return np.concatenate(preds, axis=1)


      # preds = [models[x].get_latent_space(X) for x in mn]
      # model_counts = [float(len(fl_dataY[x])) for x in mn]
      # model_counts = np.array(model_counts)
      # model_counts = model_counts/sum(model_counts)
      # preds = np.mean([np.array(p)*mc for p,mc in zip(preds, model_counts)], axis=0)
      # return preds

    def get_concat_pca(X, mn):
      preds = [models[x].transform(X) for x in mn]
      return np.concatenate(preds, axis=1)

      # preds = [models[x].transform(X) for x in mn]
      # model_counts = [float(len(fl_dataY[x])) for x in mn]
      # model_counts = np.array(model_counts)
      # model_counts = model_counts/sum(model_counts)
      # preds = sum([np.array(p)*mc for p,mc in zip(preds, model_counts)])
      # return preds

    if rep_method == 'avg_autoencoders':
      transformed_dataX = avg_model.predict(dataX_test).astype(np.float32)
    if rep_method == 'aggregate_autoencoders':
      transformed_dataX = get_concat_autoencoder(dataX_test, models_names).astype(np.float32)
    if rep_method == 'pca':
      transformed_dataX = get_concat_pca(dataX_test, models_names).astype(np.float32)
      # print("AAAAAAAAAA")
    if rep_method is None:
      transformed_dataX = dataX_test
      # print(avg_model.predict(dataX_test).astype(np.float32).shape)
      # exit()
    
    np.savetxt('dataX_test.txt', dataX_test, fmt='%f')
    np.savetxt('transformed_dataX.txt', transformed_dataX, fmt='%f')
    np.savetxt('dataY_test_nonsoft.txt', dataY_test_nonsoft, fmt='%s')
    
    clf_models = {}
    # '''
    for x in models_names:
      new_clf = RandomForestClassifier(max_depth=6)

      if (not src_ip_idx is None) and (not dst_ip_idx is None):
        train_dataX = np.delete(fl_dataX[x], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      else:
        train_dataX = fl_dataX[x].astype(np.float32)
            
      # tmp_clf = ExtraTreesClassifier(n_estimators=50)
      tmp_clf = ExtraTreesClassifier(n_estimators=250)

      # tmp_clf = tmp_clf.fit(train_dataX, le.transform(fl_dataY[x]))
      # models_fs[x] = SelectFromModel(tmp_clf, prefit=True, max_features=16)
      # models_fs[x] = SelectFromModel(tmp_clf, prefit=False)
      models_fs[x] = SelectFromModel(tmp_clf, prefit=False, max_features=8)

      if rep_method == 'avg_autoencoders':
        transformed_train_dataX = avg_model.predict(train_dataX).astype(np.float32)
        # clf_models[x] = make_pipeline(models_fs[x], new_clf)
        clf_models[x] = new_clf
      if rep_method == 'aggregate_autoencoders':
        transformed_train_dataX = get_concat_autoencoder(train_dataX, models_names).astype(np.float32)
        clf_models[x] = make_pipeline(models_fs[x], new_clf)
        # clf_models[x] = new_clf
      if rep_method == 'pca':
        transformed_train_dataX = get_concat_pca(train_dataX, models_names).astype(np.float32)
        # clf_models[x] = make_pipeline(models_fs[x], new_clf)
        clf_models[x] = new_clf
      if rep_method is None:
        transformed_train_dataX = train_dataX
        clf_models[x] = new_clf

      # transformed_train_dataX = np.delete(fl_dataX[x], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32).astype(np.float32)
      
      # new_clf.fit(transformed_train_dataX, le.transform(fl_dataY[x]))
      # clf_models[x] = new_clf
      
      
      clf_models[x].fit(transformed_train_dataX, le.transform(fl_dataY[x]))
      # clf_models[x].fit(transformed_train_dataX, fl_dataY[x])

    clf_avg_model = VotingClassifier(estimators=[(x, clf_models[x]) for x in models_names], voting='hard')
    clf_avg_model.estimators_ = [clf_models[x] for x in models_names]
    clf_avg_model.le_ = le
    clf_avg_model.classes_ = clf_avg_model.le_.classes_
    # '''


    '''
    total_transformed_train_dataX = []
    total_transformed_train_dataY = []
    for x in models_names:
      train_dataX = np.delete(fl_dataX[x], np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)

      if rep_method == 'avg_autoencoders':
        transformed_train_dataX = avg_model.predict(train_dataX).astype(np.float32)
      if rep_method == 'aggregate_autoencoders':
        transformed_train_dataX = get_concat_autoencoder(train_dataX, models_names).astype(np.float32)
      if rep_method is None:
        transformed_train_dataX = train_dataX

      total_transformed_train_dataX.append(transformed_train_dataX)
      # total_transformed_train_dataY.append(le.transform(fl_dataY[x]))
      total_transformed_train_dataY.append(fl_dataY[x])
    
    total_transformed_train_dataX = np.concatenate(total_transformed_train_dataX)
    total_transformed_train_dataY = np.concatenate(total_transformed_train_dataY)

    clf_avg_model = RandomForestClassifier(max_depth=6)
    clf_avg_model.fit(total_transformed_train_dataX, total_transformed_train_dataY)
    '''

    all_C = list(set_y)
    C = [set(fl_dataY[x]) for x in models_names]
    TC = [len(x) for x in C]

    TCW = []
    TCW_norm = {}
    for x in models_names:
      TCW.append({})
      unique, counts = np.unique(fl_dataY[x], return_counts=True)
      count_dict = dict(zip(unique, counts))
      for c in all_C:
        if not c in TCW:
          TCW[-1][c] = 0

        if c in count_dict:
          TCW[-1][c] += count_dict[c]
          if c in TCW_norm:
            TCW_norm[c] += count_dict[c]
          else:
            TCW_norm[c] = count_dict[c]
        
      
    TA = [TCW_norm[c] if c in TCW_norm else np.inf for c in all_C] # Fix


    clf_preds = [clf_models[x].predict(transformed_dataX) for x in models_names]
    predictions = []
    for i in range(len(dataX_test)):
      votes = np.array(le.inverse_transform([x[i] for x in clf_preds]))
      TV = np.zeros(len(all_C))
      for i, c in enumerate(all_C):
        for v, w in zip(votes, TCW):
          if v == c:
            TV[i] += w[c]
      TP = [TV[i]/TA[i] for i in range(len(all_C))]

      predictions.append(all_C[np.argmax(TP)])

    # print(predictions)
    # print(dataY_test_nonsoft)
    acc = sum(np.array(predictions) == dataY_test_nonsoft)/len(dataY_test)
    cm = confusion_matrix(dataY_test_nonsoft, predictions, labels=le.classes_)

    # print(set(le.classes_))
    # print(set_y)
    # print(set(dataY_test_nonsoft))
    print(cm)
    # print(classification_report(dataY_test_nonsoft, predictions, target_names=le.classes_))
    # print(classification_report(dataY_test_nonsoft, predictions, target_names=list(set_y)))
    # print(classification_report(dataY_test_nonsoft, predictions, target_names=list(set(predictions))))
    print(classification_report(dataY_test_nonsoft, predictions))
    final_y_true = dataY_test_nonsoft
    final_y_pred = predictions

    '''    
    # acc = avg_model.score(transformed_dataX, dataY_test)
    
    y_pred = clf_avg_model.predict(transformed_dataX)

    # dataY_test_nonsoft = np.array(dataY_test)
    
    # print(dataY_test_nonsoft)
    # print(np.array(y_pred))

    dataY_test_nonsoft = np.argmax(dataY_test, axis=1)
    dataY_test_nonsoft = le.inverse_transform(dataY_test_nonsoft)


    acc = sum(np.array(y_pred) == dataY_test_nonsoft)/len(dataY_test)

    cm = confusion_matrix(dataY_test_nonsoft, y_pred, labels=le.classes_)
    print(classification_report(dataY_test_nonsoft, np.array(y_pred), target_names=le.classes_))
    final_y_true = dataY_test_nonsoft
    final_y_pred = np.array(y_pred)


    unique, counts = np.unique(final_y_true, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)
    
    unique, counts = np.unique(final_y_pred, return_counts=True)
    count_dict = dict(zip(unique, counts))
    print(count_dict)
    '''
  
  if not use_dl_model and use_partial_fit:
    total_train_samples = sum([len(fl_dataX[x]) for x in fl_dataX])
    avg_contribution = {x: float(len(fl_dataX[x]))/total_train_samples for x in fl_dataX}
    
    avg_weights = sum([np.array(models[x].coefs_) for x in models])/len(models)
    avg_model.coefs_ = avg_weights

    acc = avg_model.score(dataX_test, dataY_test)

    
  print(f"Final Result (Accuracy macro): {acc}")
  print_method_result_metrics(final_y_true, final_y_pred)

  # continue
  return 


      # print(dataX)
      # print(ip, len(dataX))
  
  # exit()
  # weights = np.zeros((total_ips, total_ips))

  # for i in range(total_ips):
  #   for j in range(i+1, total_ips):
  #     ip1 = ip_set[i]
  #     ip2 = ip_set[j]
      
  #     w = np.sum(np.where((total_data_X[:, src_ip_idx] == ip1 & total_data_X[:, src_ip_idx] == ip2) | (total_data_X[src_ip_idx] == ip2 & total_data_X[src_ip_idx] == ip1) ))
  #     weights[i,j] = w
  #     weights[j,i] = w
    
  # print(total_data_X.shape)
  # print(weights)
  # print(total_data_X.shape)
  # print(ip_set)
  # print(total_ips)
  

  # g.add_nodes_from(list(set(src_ip_list).union(set(dst_ip_list))))
  plot_network_traffic = False

  if plot_network_traffic:
    g = nx.Graph()

    le = LabelEncoder()
    le.fit(ip_set)

    src_ip_list = le.transform(total_data_X[:, src_ip_idx])
    dst_ip_list = le.transform(total_data_X[:, dst_ip_idx])
    
    import tqdm
    import pylab

    ip_idx_bot = []
    ip_idx_to_color = {}

    graph_weights = {}
    for ip1, ip2, ip1_n, ip2_n, y in tqdm.tqdm(zip(src_ip_list, dst_ip_list, total_data_X[:, src_ip_idx], total_data_X[:, dst_ip_idx], total_data_Y), total=len(src_ip_list)):
      
      if is_ip_private(ip1_n) and is_ip_private(ip2_n):
        # g.add_edge(ip1, ip2)

        # g.add_edge(ip1_n, ip2_n)

        if not ip1_n in graph_weights:
          graph_weights[ip1_n] = {}
        if not ip2_n in graph_weights:
          graph_weights[ip2_n] = {}
        
        if not ip2_n in graph_weights[ip1_n]:
          graph_weights[ip1_n][ip2_n] = 0
        if not ip1_n in graph_weights[ip2_n]:
          graph_weights[ip2_n][ip1_n] = 0

        graph_weights[ip1_n][ip2_n]+=1
        graph_weights[ip2_n][ip1_n]+=1

        # print(ip1_n, ip2_n)

      else:
        if not is_ip_private(ip1_n) and is_ip_private(ip2_n):
          # g.add_edge('Internet', ip2_n)
          # print(ip2_n)
          ip1_n = 'Internet'

          if not ip1_n in graph_weights:
            graph_weights[ip1_n] = {}
          if not ip2_n in graph_weights:
            graph_weights[ip2_n] = {}
          
          if not ip2_n in graph_weights[ip1_n]:
            graph_weights[ip1_n][ip2_n] = 0
          if not ip1_n in graph_weights[ip2_n]:
            graph_weights[ip2_n][ip1_n] = 0

          graph_weights[ip1_n][ip2_n]+=1
          graph_weights[ip2_n][ip1_n]+=1

        if not is_ip_private(ip2_n) and is_ip_private(ip1_n):
          # g.add_edge(ip1_n, 'Internet')
          # print(ip1_n)
          ip2_n = 'Internet'
        
          if not ip1_n in graph_weights:
            graph_weights[ip1_n] = {}
          if not ip2_n in graph_weights:
            graph_weights[ip2_n] = {}
          
          if not ip2_n in graph_weights[ip1_n]:
            graph_weights[ip1_n][ip2_n] = 0
          if not ip1_n in graph_weights[ip2_n]:
            graph_weights[ip2_n][ip1_n] = 0

          graph_weights[ip1_n][ip2_n]+=1
          graph_weights[ip2_n][ip1_n]+=1

      # print(y)
      # if y != 'normal':
      
      
      if y == 'ddos':
        ip_idx_to_color[ip1_n] = 'red'
    

    for ip1_n in graph_weights:
      for ip2_n in graph_weights[ip1_n]:
        w = graph_weights[ip1_n][ip2_n]
        w = np.log2(w)
        g.add_edge(ip1_n, ip2_n, weight=w)
        # print(graph_weights[ip1_n][ip2_n])
    
    ip_idx_to_color['Internet'] = 'yellow'
    color_map = []
    for node in g:
      if node in ip_idx_to_color:
        color_map.append(ip_idx_to_color[node])
      else:
        color_map.append('blue')
    
    # color_map = ['blue']*len(g)
    # for ip in ip_idx_bot:
    #   color_map[ip] = 'red'

    # print(len(g))
    # print(len([x for x in color_map if x == 'green']))

    '''
    pos = nx.spring_layout(g)
    # pos = nx.spring_layout(g, k=0.5, iterations=1, scale=3)
    # pos = nx.spring_layout(g, k=1000)
    # pos = nx.spring_layout(g, scale=10, k=0.15, iterations=20)  # default to scale=1
    # pos = nx.spring_layout(g, iterations=200, k=2)  # default to scale=1
    nx.draw(g, pos, node_color=color_map, with_labels=True, font_size=4, font_color='green', node_size=40)
    '''

    edges,weights = zip(*nx.get_edge_attributes(g,'weight').items())

    pos = nx.spring_layout(g)
    # nx.draw(g, pos, node_color='b', edgelist=edges, edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)

    d = dict(g.degree)
    # nx.draw(g, nodelist=d.keys(), node_size=[v * 100 for v in d.values()])
    nx.draw(g, pos, node_color=color_map, with_labels=True, font_size=4,
            edge_color=weights, font_color='green',
            # node_size=40,
            node_size=[v * 10 for v in d.values()]
            # )
            ,edge_cmap=plt.cm.Blues)
    # plt.savefig('edges.png')



    # nx.draw(g, pos, node_color=color_map, font_size=8, font_color='red', node_size=60)
    # nx.draw_graphviz(G, 'neato')
    # nx.draw(g, pos=nx.drawing.nx_agraph.graphviz_layout(g), node_size=1600, cmap=plt.cm.Blues,
    #     node_color=range(len(g)))

    pylab.figure(1,figsize=(50,50))
    # pylab.xlim(0,1)
    # pylab.ylim(0,1)
    #nx.draw(G,pos,font_size=10)
    # pylab.show()
    pylab.savefig("filename.pdf")
    exit()


  

  
  # proc_and_vis(total_data_X, total_data_Y,
  # dim_red_type='select_kbest_chi',
  # dim_red_size=16,
  # norm_type='stdscaller',
  # vis_type='tsne',
  # fig_name=f'{ds_name}_vis_tsne_select_kbest_chi.pdf')

  # proc_and_vis(total_data_X, total_data_Y,
  # dim_red_type='pca',
  # dim_red_size=16,
  # norm_type='stdscaller',
  # vis_type='tsne',
  # fig_name=f'{ds_name}_vis_tsne_pca.pdf')

  # proc_and_vis(total_data_X, total_data_Y,
  # dim_red_type='select_kbest_chi',
  # dim_red_size=16,
  # norm_type='stdscaller',
  # vis_type='pca',
  # fig_name=f'{ds_name}_vis_pca_select_kbest_chi.pdf')

  # proc_and_vis(total_data_X, total_data_Y,
  # dim_red_type='select_kbest_chi',
  # dim_red_size=16,
  # norm_type='stdscaller',
  # vis_type='pca',
  # fig_name=f'{ds_name}_vis_pca_pca.pdf')

  proc_and_vis(total_data_X, total_data_Y,
  dim_red_type='pca',
  dim_red_size='mle',
  norm_type='stdscaller',
  vis_type='tsne',
  fig_name=f'{ds_name}_vis_tsne_pca.png')
  # fig_name=f'{ds_name}_vis_tsne_pca.pdf')
  # exit()
  return
  
  
  skf = StratifiedKFold(n_splits=10)
  # skf = StratifiedKFold(n_splits=2)

  fold_id = 0
  cm_list = []
  cr_list = []
  aucroc_list = []

  fi_list = {x: [] for x in feature_names}

  for train_index, test_index in skf.split(total_data_X, total_data_Y):
    
    # print("TRAIN:", train_index, "TEST:", test_index)
    trainX, testX = total_data_X[train_index], total_data_X[test_index]
    trainY, testY = total_data_Y[train_index], total_data_Y[test_index]

    # scaler = preprocessing.StandardScaler().fit(trainX)
    scaler = preprocessing.StandardScaler().fit(np.concatenate([trainX,testX]))
    trainX = scaler.transform(trainX)
    testX = scaler.transform(testX)
    

    # trainX = trainX[:1000, :]
    # testX = testX[:1000, :]
    # trainY = trainY[:1000]
    # testY = testY[:1000]

    # print(trainY.shape)

    random_state = np.random.RandomState(0)
    # classifier = OneVsRestClassifier(
    #     # svm.SVC(kernel="linear", probability=True, random_state=random_state)
    #     RandomForestClassifier(random_state=0)
    # )

    

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.datasets import load_iris
    from sklearn.feature_selection import SelectFromModel


    # f_selector = SelectKBest(chi2, k=16)
    # f_selector.fit(trainX, trainY)

    from sklearn.kernel_approximation import RBFSampler
    
    start_time = time.time()
    # class_red_type = 'rbf'
    

    if class_red_type == 'rbf':
      # f_selector = RBFSampler(gamma=1, random_state=1, n_components=16)
      f_selector = RBFSampler(random_state=1)
      f_selector.fit(trainX)

    if class_red_type == 'etc_select':
      clf = ExtraTreesClassifier(n_estimators=16)
      clf = clf.fit(trainX, trainY)
      # f_selector = SelectFromModel(clf, prefit=True)
      f_selector = SelectFromModel(clf, prefit=True, max_features=32, threshold=-np.inf)

    if class_red_type == 'pca':
      f_selector = PCA(n_components='mle')
      f_selector.fit(trainX)

    if class_red_type != 'etc_select':
      clf = ExtraTreesClassifier(n_estimators=16)
      clf = clf.fit(trainX, trainY)
    
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    
    for fi, fi_std, fn in zip(importances, std, feature_names):
      fi_list[fn].append(fi)
    
    # print(fi_list)
    # exit()
    
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")



    # '''
    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    # forest_importances.plot.bar(yerr=std, ax=ax)
    forest_importances.plot.bar(yerr=std, ax=ax, rot=80, fontsize=7)
    ax.set_title("Importâncias de características usando MDI")
    ax.set_ylabel("Mean decrease in impurity (MDI)")
    # ax.set_xticklabels(rotation = 45)
    fig.tight_layout()
    fig.savefig(f'f_import_{ds_name}_fold_{fold_id}.pdf')
    # '''
    exit()

    # from sklearn.feature_selection import RFE
    # from sklearn.svm import SVR
    # estimator = SVR(kernel="linear")
    # f_selector = RFE(estimator, n_features_to_select=16, step=8)
    # f_selector = f_selector.fit(trainX[:10000], trainY[:10000])

    
    start_time = time.time()
    if use_dimred:
      trainX = f_selector.transform(trainX)
      testX = f_selector.transform(testX)
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to select dim: {elapsed_time:.3f} seconds")



    # clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf = RandomForestClassifier(random_state=0)
    start_time = time.time()
    clf.fit(trainX, trainY)
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to fit the model: {elapsed_time:.3f} seconds")

    # print(clf.score(testX, testY))

    start_time = time.time()
    predictions = clf.predict(testX)
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to predict: {elapsed_time:.3f} seconds")

    start_time = time.time()
    class_list = ['normal']+list(set(clf.classes_)-{'normal'})
    cm = confusion_matrix(testY, predictions, labels=class_list)
    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to create confusion matrix: {elapsed_time:.3f} seconds")

    '''
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                            display_labels=clf.classes_)
    disp.plot()
    # plt.show()
    plt.savefig(f'cm_{ds_name}_fold_{fold_id}.pdf')
    '''
    
    fold_id += 1

    cm_list.append(cm)

    # print(trainX)
    # print(trainY)
    # print(testX)
    # print(testY)


    start_time = time.time()
    y_true = testY
    y_pred = predictions

    cr = classification_report(y_true, y_pred, output_dict=True)
    cr_list.append(cr)
    # print(cr)

    # classifier = RandomForestClassifier(random_state=0)
    classifier = clf

    # trainY_bin = np.zeros((len(trainY), 2))
    # testY_bin = np.zeros((len(testY), 2))

    # trainY_bin[trainY != 'normal', 1] = 1
    # testY_bin[testY != 'normal', 1] = 1
    # trainY_bin[trainY == 'normal', 0] = 1
    # testY_bin[testY == 'normal', 0] = 1

    # create_roc_curve_plot(classifier, trainX, testX, trainY_bin, testY_bin,
    roc_auc = create_roc_curve_plot(classifier, trainX, testX, trainY, testY,
      figname=f'roccurve_{ds_name}_{fold_id}.pdf', legend=f"{class_type}")

    elapsed_time = time.time() - start_time
    # print(f"Elapsed time to compute ROC curve: {elapsed_time:.3f} seconds")
    
    aucroc_list.append(roc_auc)

  start_time = time.time()
  importances = [np.mean(fi_list[fn]) for fn in feature_names]
  std = [np.std(fi_list[fn]) for fn in feature_names]

  top_features = np.array(feature_names)[np.argsort(importances)[-5:][::-1]]
  print(f"{attack_type}: {top_features}")
  
  elapsed_time = time.time() - start_time

  # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

  forest_importances = pd.Series(importances, index=feature_names)
  
  
  # '''
  fig, ax = plt.subplots()
  forest_importances.plot.bar(yerr=std, ax=ax)
  ax.set_title("Feature importances using MDI")
  ax.set_ylabel("Mean decrease in impurity")
  fig.tight_layout()
  fig.savefig(f'f_import_{ds_name}.pdf')
  # '''
  
  # cm_mean = np.mean(cm_list, axis=0)
  cm_sum = np.sum(cm_list, axis=0)

  disp = ConfusionMatrixDisplay(confusion_matrix=cm_sum,
                            display_labels=clf.classes_)
  disp.plot()
  # plt.show()
  plt.savefig(f'cm_{ds_name}_sum.pdf')
  # fig.savefig(f'cm_{ds_name}_sum.pdf')


  # acc_macro_mean = np.sum([x['accuracy'] for x in cr_list], axis=0)
  # acc_micro_mean = np.sum([x['accuracy'] for x in cr_list], axis=0)

  n_folds = len(cr_list)

  acc_mean = np.sum([x['accuracy'] for x in cr_list], axis=0)/n_folds

  prec_macro_mean = np.sum([x['macro avg']['precision'] for x in cr_list], axis=0)/n_folds
  prec_micro_mean = np.sum([x['weighted avg']['precision'] for x in cr_list], axis=0)/n_folds

  recall_macro_mean = np.sum([x['macro avg']['recall'] for x in cr_list], axis=0)/n_folds
  recall_micro_mean = np.sum([x['weighted avg']['recall'] for x in cr_list], axis=0)/n_folds

  f1scr_macro_mean = np.sum([x['macro avg']['f1-score'] for x in cr_list], axis=0)/n_folds
  f1scr_micro_mean = np.sum([x['weighted avg']['f1-score'] for x in cr_list], axis=0)/n_folds

  aucroc_mean = np.sum(aucroc_list)/n_folds
  # print(f"CM Avg:\n{cm_mean}")
  print(f"Acc Avg: {acc_mean}")
  print(f"AUC ROC Avg: {aucroc_mean}")

  print(f"CM Sum:\n{cm_sum}")
  
  print(f"Prec Macro-Avg: {prec_macro_mean}")
  print(f"Prec Weighted-Avg: {prec_micro_mean}")

  print(f"Recall Macro-Avg: {recall_macro_mean}")
  print(f"Recall Weighted-Avg: {recall_micro_mean}")

  print(f"F1-Score Macro-Avg: {f1scr_macro_mean}")
  print(f"F1-Score Weighted-Avg: {f1scr_micro_mean}")
  
  result_data.append({
    'ds_name': ds_name,
    'args': ton_args,
    'results': {
      'acc_mean': acc_mean,
      'aucroc_mean': aucroc_mean,
      'cm_sum': cm_sum.tolist(),
      'prec_macro_mean': prec_macro_mean,
      'prec_micro_mean': prec_micro_mean,
      'recall_macro_mean': recall_macro_mean,
      'recall_micro_mean': recall_micro_mean,
      'f1scr_macro_mean': f1scr_macro_mean,
      'f1scr_micro_mean': f1scr_micro_mean,
    },
  })

  exit()
  with open(f'{args.output_file}', 'w') as f:
    json.dump(result_data, f)


if __name__ == "__main__":
  main()