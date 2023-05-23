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
        {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
      ]
    
    if data_set_name == 'botiot':
      arg_test = [
        {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': test_ratio , 'drop_cols': ['sport', 'dport', 'category', 'subcategory', 'pkSeqID']},
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
    else:
      trainX = trainX.astype(np.float32)
      testX = testX.astype(np.float32)


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
    return
      


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()


  

if __name__ == "__main__":
  main()