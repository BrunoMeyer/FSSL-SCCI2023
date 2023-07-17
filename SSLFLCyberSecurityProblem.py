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

from SSLFLProblem import SSLFLProblem

import copy

ACEPTED_DS_NAMES = [
  'botiot',
  'toniot',
  'nsl-kdd',
]
class SSLFLCyberSecurityProblem(SSLFLProblem):
  def __init__(self, input_file, data_set_name='botiot', test_ratio=0.01, n_clients=10, dirichlet_beta=0.1, random_seed=0):
    self.data_set_name = data_set_name
    self.test_ratio = test_ratio
    self.n_clients = n_clients
    self.dirichlet_beta = dirichlet_beta
    self.random_seed = random_seed

    if not (data_set_name in ACEPTED_DS_NAMES):
      raise '"{}" dataset is not available. Please choose one of the following datasets: {}'.format(data_set_name, ACEPTED_DS_NAMES)
    

    self.data_set_name = data_set_name

    if data_set_name == 'toniot':
      arg_test = [
        # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
        {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['type', 'label', 'ts']},
      ]
    
    if data_set_name == 'botiot':
      arg_test = [
        # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},
        {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['category', 'subcategory', 'pkSeqID']},
      ]
    
    if data_set_name == 'nsl-kdd':
      arg_test = [
        {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},
      ]
    
    # n_clients = 50
    # n_clients = 10
    # random_seed = 0


    
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

    ton_args_copy = ton_args.copy()

    if 'use_dimred' in ton_args_copy:
      del ton_args_copy['use_dimred']
    if 'class_red_type' in ton_args_copy:
      del ton_args_copy['class_red_type']

    if data_set_name == 'toniot':
      trainX, trainY, testX, testY, feature_names, ds_name = load_toniot(input_file, **ton_args_copy)
      test_size_rate = 0.0009
    
    if data_set_name == 'botiot':
      trainX, trainY, testX, testY, feature_names, ds_name = load_botiot(input_file, **ton_args_copy)
      test_size_rate = 0.0009


      undersample_strategy_dict = {}
      for y in set(list(testY)):
        undersample_strategy_dict[y] = int(sum(testY==y)/100)
      undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=undersample_strategy_dict, random_state=0)

      testX, testY = undersample.fit_resample(testX, testY)
      

    if data_set_name == 'nsl-kdd':
      trainX, trainY, testX, testY, feature_names, ds_name = load_nslkdd(ds_path=input_file)
      test_size_rate = 0.0009

      undersample_strategy_dict = {}
      for y in set(list(testY)):
        undersample_strategy_dict[y] = int(sum(testY==y)/4)
      undersample = imblearn.under_sampling.RandomUnderSampler(sampling_strategy=undersample_strategy_dict, random_state=0)


      testX, testY = undersample.fit_resample(testX, testY)
    
    self.feature_names = feature_names
    # print([x.dtype for x in testX.T])
    # exit()
    
    data_augmentation = None
    # dirichlet_beta = 100
    # dirichlet_beta = 0.1

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

      mask = np.ones(len(feature_names)).astype(np.bool)
      mask[[src_ip_idx, dst_ip_idx]] = False
      feature_names = np.array(feature_names)[mask]
    else:
      trainX = trainX.astype(np.float32)
      testX = testX.astype(np.float32)

    self.feature_names = feature_names

    print('Dataset name: {}'.format(data_set_name))
    print('Max trainX value: {}'.format(np.max(trainX)))
    print('Max testX value: {}'.format(np.max(testX)))
    print('Total classes train: {}'.format(len(set(list(trainY)))))
    print('Total classes test: {}'.format(len(set(list(testY)))))
    print('Total instances train: {}'.format(len(trainY)))
    print('Total instances test: {}'.format(len(testY)))
    print('Total instances: {}'.format(len(testY) + len(trainY)))
      
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
    self.le = le

    # convert_dict = {}
    # for y in set(list(testY)+list(trainY)):
    #   convert_dict[y] = le.inverse_transform([y])[0]
    
    # print(convert_dict)
    # exit()


    client_dataX, server_dataX, client_dataY, server_dataY = train_test_split(
    trainX, trainY, test_size=test_size_rate, random_state=42,
    stratify=trainY,
    shuffle=True
    )


    d = partition_data(
      (trainX, trainY, trainX, trainY),
      "",
      "logdir/",
      "noniid-labeldir", # iid-diff-quantity, noniid-labeldir
      n_clients,
      beta=dirichlet_beta,
      random_seed=random_seed
    )

    mask_total_data_X_local_ips = np.ones(trainX.shape[0], dtype=np.bool)

    
    fl_dataX = [trainX[d[4][i]] for i in d[4]]
    fl_dataY = [trainY[d[4][i]] for i in d[4]]


    SSLFLProblem.__init__(self, fl_dataX, server_dataX, server_dataY, testX, testY, clients_dataY=fl_dataY)

    if self.data_set_name == 'toniot':
      self.forbidden_features = ['src_port', 'dst_port']
      self.feature_label_candidates = ['src_port', 'dst_port']

    if self.data_set_name == 'botiot':
      self.forbidden_features = ['sport', 'dport']
      self.feature_label_candidates = ['sport', 'dport']

    self.remove_forbidden_features()
    self.create_pretext_label_candidates()
    return
  
  def create_pretext_label_candidates(self):
    f_idx = []
    for i, fn in enumerate(self.raw_feature_names):
      if fn in self.feature_label_candidates:
        f_idx.append(i)

    f_idx = np.array(f_idx)

    mask = np.zeros(len(self.raw_feature_names)).astype(np.bool)
    mask[f_idx] = True

    self.pretext_trainY = self.raw_trainX[:, mask].astype(np.int32)
    self.pretext_testY = self.raw_testX[:, mask].astype(np.int32)

    clients_pretext_dataY = []
    for i in range(len(self.raw_clients_dataX)):
      pretext_labels = self.raw_clients_dataX[i][:, mask].astype(np.int32)
      clients_pretext_dataY.append(pretext_labels)
    
    self.clients_pretext_dataY = np.array(clients_pretext_dataY)
    # print(self.clients_pretext_dataY)
    # exit()

  def remove_forbidden_features(self):
    f_idx = []
    for i, fn in enumerate(self.feature_names):
      if fn in self.forbidden_features:
        f_idx.append(i)

    f_idx = np.array(f_idx)

    self.raw_feature_names = copy.deepcopy(self.feature_names)
    self.raw_trainX = copy.deepcopy(self.trainX)
    self.raw_testX = copy.deepcopy(self.testX)
    self.raw_clients_dataX = copy.deepcopy(self.clients_dataX)

    mask = np.ones(len(self.feature_names)).astype(np.bool)
    mask[f_idx] = False
    self.trainX = self.trainX[:, mask]
    self.testX = self.testX[:, mask]

    for i in range(len(self.clients_dataX)):
      self.clients_dataX[i] = self.clients_dataX[i][:, mask]

    self.data_type = type(self.trainX[0][0])
    self.input_shape = self.trainX.shape[1]

    self.feature_names = self.feature_names[mask]

  def get_pretext_dataset(self):

    if self.data_set_name == 'toniot':
      print(list(zip(list(self.feature_names), list(self.trainX[0]))))
    if self.data_set_name == 'botiot':
      print(list(zip(list(self.feature_names), list(self.trainX[0]))))

    exit()

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()


  

if __name__ == "__main__":
  main()