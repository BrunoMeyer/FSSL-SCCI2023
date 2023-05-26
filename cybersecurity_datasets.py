import pandas as pd
import numpy as np
import argparse

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

import gc
from sklearn import preprocessing

import matplotlib
import matplotlib.pyplot as plt

from os.path import isfile, join
import pickle

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.manifold import TSNE
# from tsnecuda import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel

from plot_roc_curve_ex import create_roc_curve_plot
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
from sklearn.ensemble import RandomForestClassifier


import time

import json

import networkx as nx
 
import tensorflow as tf
# import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing


import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()




def is_ip_private(ip):
  return ip.startswith('192.168')
    

def process_toniot_df(
  df,
  class_type='multiclass',
  attack_type=None,
  return_fnames=False,
  drop_cols=[],
  test_ratio=0.33):
  print(df.columns)

  '''
  last_value = None
  total_values = []
  for i, (t, ts) in enumerate(zip(df['type'], df['ts'])):
    if t != last_value:
      last_value = t
      total_values.append((i, ts, t))

  print("\n".join([str(x) for x in total_values]))
  exit()
  '''

  # # train_df = df[:461033]
  # # test_df = df[461033:]
  # train_df = df[:200000]
  # test_df = df[200000:]

  # if class_type == 'multiclass':
  #   trainY = train_df['type'].to_numpy()
  #   testY = test_df['type'].to_numpy()
  # elif class_type == 'binary':
  #   trainY = train_df['label'].to_numpy()
  #   testY = test_df['label'].to_numpy()


  # train_df = train_df.drop(['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port'], axis = 1)
  # test_df = test_df.drop(['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port'], axis = 1)
  
  # for c in train_df.columns:
  #   if train_df[c].dtype == object:
  #     # ohe = OneHotEncoder(sparse = False)
  #     # ohe = OneHotEncoder()
  #     ohe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
  #     ohe.fit(train_df[[c]])
  #     train_df[[c]] = ohe.transform(train_df[[c]])
  #     test_df[[c]] = ohe.transform(test_df[[c]])
  #     gc.collect()
  #     print(c)
  # 
  # trainX = train_df.to_numpy()
  # testX = test_df.to_numpy()

  if class_type in ['multiclass', 'one_attack', 'one_attack_three_classes']:
    dataY = df['type'].to_numpy()
  elif class_type == 'binary':
    dataY = df['label'].to_numpy()

  # ohe = LabelEncoder()
  # ohe.fit(dataY)
  # dataY = ohe.transform(dataY)
  gc.collect()

  df = df.drop(drop_cols, axis = 1)
  # df = df.drop(['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port'], axis = 1)
  # df = df.drop(['type', 'label'], axis = 1)

  data_types = [df[c].dtype for c in df.columns]
  feature_names = list(df.columns)
  
  dataX = df.to_numpy()

  if class_type == 'one_attack':
    if attack_type != 'normal':
      mask_train = np.logical_or(dataY == attack_type, dataY == 'normal')
      dataX = dataX[mask_train]
      dataY = dataY[mask_train]
    else:
      dataY[dataY != 'normal'] = 'attack'
  if class_type == 'one_attack_three_classes':
    if attack_type == 'normal':
      raise Exception("Normal attack not allowed using class type 'one_attack_three_classes'")
    mask_train = np.logical_and(dataY != attack_type, dataY != 'normal')
    dataY[mask_train] = 'Other'
      

  # '''
  X_train, X_test, y_train, y_test = train_test_split(
   dataX, dataY, test_size=test_ratio, random_state=42,
   stratify=dataY
    # shuffle=False
   )
  # '''


  '''
  unique, counts = np.unique(dataY, return_counts=True)
  count_dict = dict(zip(unique, counts))
  total_train_idx, total_test_idx = [], []

  for y in count_dict:
    train_size = int((1 - test_ratio)*count_dict[y])
    train_idx = np.where(dataY == y)[0][:train_size]
    test_idx = np.where(dataY == y)[0][train_size:]
    total_train_idx.append(train_idx)
    total_test_idx.append(test_idx)

  total_train_idx = np.concatenate(total_train_idx)
  total_test_idx = np.concatenate(total_test_idx)
  X_train = dataX[total_train_idx]
  X_test = dataX[total_test_idx]
  y_train = dataY[total_train_idx]
  y_test = dataY[total_test_idx]
  '''

  for c in range(X_train.shape[1]):
    print(c, feature_names[c], data_types[c])
    # if X_train[:, c].dtype == object:
    # if data_types[c] == object:
    if data_types[c] == object and not feature_names[c] in ['src_ip', 'dst_ip']:
      # ohe = OneHotEncoder(sparse = False)
      # ohe = OneHotEncoder()
      ohe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
      ohe.fit(X_train[:, [c]])
      X_train[:, [c]] = ohe.transform(X_train[:, [c]])
      X_test[:, [c]] = ohe.transform(X_test[:, [c]])
      gc.collect()

  # ohe = LabelEncoder()
  # ohe.fit(trainY)
  # trainY = ohe.transform(trainY)
  # testY = ohe.transform(testY)
  # gc.collect()

  print(X_train.shape)
  print(y_train.shape)
  print(X_test.shape)
  print(y_test.shape)

  # for i in range(trainX.shape[1]):
  #   print(trainX[0, i].dtype)

  if return_fnames:
    return X_train, y_train, X_test, y_test, feature_names
  else:
    return X_train, y_train, X_test, y_test

def load_toniot(
  input_file,
  class_type='binary',
  attack_type=None,
  return_fnames=True,
  drop_cols=[],
  test_ratio=0.33):
  
  if len(drop_cols) > 0:
    drop_col_names = 'drop_' + '_'.join(drop_cols)
  else:
    drop_col_names = ''
  
  ret_fnames_name = 'return_fnames' if return_fnames else ''


  if class_type != 'one_attack' and class_type != 'one_attack_three_classes':
    ds_name = f'toniot_{class_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
  else:
    ds_name = f'toniot_{class_type}_{attack_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
    
  cache_path = f'cache/toniot_{ds_name}.pickle'

  if isfile(cache_path):
    with open(cache_path, 'rb') as handle:
      data = pickle.load(handle)
    return (*data, ds_name)
  
  print(input_file)
  df = pd.read_csv(input_file)

          
  if  'Train_Test_Network.csv' in input_file:
    # trainX, trainY, testX, testY = process_toniot_df(df)
    # trainX, trainY, testX, testY = process_toniot_df(df, class_type='binary')
    data = process_toniot_df(
      df,
      class_type=class_type,
      attack_type=attack_type,
      return_fnames=return_fnames,
      drop_cols=drop_cols,
      test_ratio=test_ratio)

  with open(cache_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return (*data, ds_name)


# def load_nslkdd(ds_path='datasets/nsl-kdd', return_fnames=True, diff_level=10):
def load_nslkdd(ds_path='datasets/nsl-kdd', return_fnames=True, diff_level=-1):

  d_train = np.loadtxt(f"{ds_path}/KDDTrain+.txt", delimiter=',', dtype=object)
  y_train = d_train[:, -2]
  trainY_diff = d_train[:, -1].astype(np.int32)
  X_train = d_train[:, :-2]


  d_test = np.loadtxt(f"{ds_path}/KDDTest+.txt", delimiter=',', dtype=object)
  y_test = d_test[:, -2]
  testY_diff = d_test[:, -1].astype(np.int32)
  X_test = d_test[:, :-2]


  if diff_level > 0:
    train_mask = np.where(trainY_diff > 21-diff_level)[0]
    test_mask = np.where(testY_diff > 21-diff_level)[0]

    y_train = y_train[train_mask]
    trainY_diff = trainY_diff[train_mask]
    X_train = X_train[train_mask]

    y_test = y_test[test_mask]
    testY_diff = testY_diff[test_mask]
    X_test = X_test[test_mask]

  # print(len(X_train))
  # print(len(X_test))
  # exit()

  for c in range(X_train.shape[1]):
    try:
      X_train[:, c] = X_test[:, c].astype(np.float)
      X_train[:, c] = X_test[:, c].astype(np.float)
      # ohe = OneHotEncoder(sparse = False)
      # ohe = OneHotEncoder()
    except:
      ohe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
      ohe.fit(X_train[:, [c]])
      X_train[:, [c]] = ohe.transform(X_train[:, [c]])
      X_test[:, [c]] = ohe.transform(X_test[:, [c]])
      gc.collect()


  ds_name = 'nslkdd_default'
  if return_fnames:
    feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]
    return X_train, y_train, X_test, y_test, feature_names, ds_name
  else:
    return X_train, y_train, X_test, y_test, ds_name

def process_botiot_df(
  df_train,
  df_test,
  class_type='multiclass',
  attack_type=None,
  return_fnames=False,
  drop_cols=[],
  test_ratio=0.33):
  print(df_train.columns)

  '''
  last_value = None
  total_values = []
  for i, (t, ts) in enumerate(zip(df['type'], df['ts'])):
    if t != last_value:
      last_value = t
      total_values.append((i, ts, t))

  print("\n".join([str(x) for x in total_values]))
  exit()
  '''

  if class_type in ['multiclass', 'one_attack', 'one_attack_three_classes']:
    dataY_train = df_train['category'].to_numpy()
    dataY_test = df_test['category'].to_numpy()
  elif class_type == 'binary':
    dataY_train = df_train['subcategory'].to_numpy()
    dataY_test = df_test['subcategory'].to_numpy()

  # ohe = LabelEncoder()
  # ohe.fit(dataY)
  # dataY = ohe.transform(dataY)
  gc.collect()

  # print(df_train.columns)
  df_train = df_train.drop(drop_cols, axis = 1)
  df_test = df_test.drop(drop_cols, axis = 1)
  # df = df.drop(['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port'], axis = 1)
  # df = df.drop(['type', 'label'], axis = 1)

  data_types = [df_train[c].dtype for c in df_train.columns]
  feature_names = list(df_train.columns)
  
  dataX_train = df_train.to_numpy()
  dataX_test = df_test.to_numpy()

  if class_type == 'one_attack':
    if attack_type != 'Normal':
      mask_train = np.logical_or(dataY_train == attack_type, dataY_train == 'Normal')
      dataX_train = dataX_train[mask_train]
      dataY_train = dataY_train[mask_train]
      
      mask_train = np.logical_or(dataY_test == attack_type, dataY_test == 'Normal')
      dataX_test = dataX_test[mask_train]
      dataY_test = dataY_test[mask_train]
      
    else:
      dataY_train[dataY_train != 'Normal'] = 'attack'
      dataY_test[dataY_test != 'Normal'] = 'attack'
  if class_type == 'one_attack_three_classes':
    if attack_type == 'Normal':
      raise Exception("Normal attack not allowed using class type 'one_attack_three_classes'")
    mask_train = np.logical_and(dataY_train != attack_type, dataY_train != 'Normal')
    dataY_train[mask_train] = 'Other'
    mask_test = np.logical_and(dataY_test != attack_type, dataY_test != 'Normal')
    dataY_test[mask_train] = 'Other'
      

  # '''
  X_train, X_test, y_train, y_test = dataX_train, dataX_test, dataY_train, dataY_test


  '''
  unique, counts = np.unique(dataY, return_counts=True)
  count_dict = dict(zip(unique, counts))
  total_train_idx, total_test_idx = [], []

  for y in count_dict:
    train_size = int((1 - test_ratio)*count_dict[y])
    train_idx = np.where(dataY == y)[0][:train_size]
    test_idx = np.where(dataY == y)[0][train_size:]
    total_train_idx.append(train_idx)
    total_test_idx.append(test_idx)

  total_train_idx = np.concatenate(total_train_idx)
  total_test_idx = np.concatenate(total_test_idx)
  X_train = dataX[total_train_idx]
  X_test = dataX[total_test_idx]
  y_train = dataY[total_train_idx]
  y_test = dataY[total_test_idx]
  '''

  for c in range(X_train.shape[1]):
    print(c, feature_names[c], data_types[c])
    # if X_train[:, c].dtype == object:
    # if data_types[c] == object:
    if data_types[c] == object and not feature_names[c] in ['saddr', 'daddr']:
      # ohe = OneHotEncoder(sparse = False)
      # ohe = OneHotEncoder()
      ohe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
      ohe.fit(X_train[:, [c]])
      X_train[:, [c]] = ohe.transform(X_train[:, [c]])
      X_test[:, [c]] = ohe.transform(X_test[:, [c]])
      gc.collect()

  # ohe = LabelEncoder()
  # ohe.fit(trainY)
  # trainY = ohe.transform(trainY)
  # testY = ohe.transform(testY)
  # gc.collect()

  print(X_train.shape)
  print(y_train.shape)
  print(X_test.shape)
  print(y_test.shape)

  # for i in range(trainX.shape[1]):
  #   print(trainX[0, i].dtype)

  if return_fnames:
    return X_train, y_train, X_test, y_test, feature_names
  else:
    return X_train, y_train, X_test, y_test

def load_botiot(
  input_path,
  class_type='binary',
  attack_type=None,
  return_fnames=True,
  drop_cols=[],
  test_ratio=0.33):
  
  if len(drop_cols) > 0:
    drop_col_names = 'drop_' + '_'.join(drop_cols)
  else:
    drop_col_names = ''
  
  ret_fnames_name = 'return_fnames' if return_fnames else ''


  if class_type != 'one_attack' and class_type != 'one_attack_three_classes':
    ds_name = f'botiot_{class_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
  else:
    ds_name = f'botiot_{class_type}_{attack_type}_{drop_col_names}_test_{test_ratio}_{ret_fnames_name}'
    
  cache_path = f'cache/botiot_{ds_name}.pickle'

  if isfile(cache_path):
    with open(cache_path, 'rb') as handle:
      data = pickle.load(handle)
    return (*data, ds_name)


  df_train = pd.read_csv(f'{input_path}/UNSW_2018_IoT_Botnet_Final_10_best_Training.csv')
  df_test = pd.read_csv(f'{input_path}/UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv')
  # print(df)
  # exit()
  

          
  
  data = process_botiot_df(
    df_train,
    df_test,
    class_type=class_type,
    attack_type=attack_type,
    return_fnames=return_fnames,
    drop_cols=drop_cols,
    test_ratio=test_ratio)

  with open(cache_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

  return (*data, ds_name)





def random_motion(points, random_motion_force):
  """Move each point with small value
      Parameters
      ----------
      random_state: int
          The seed used to construct the trees. After the construction of each
          tree, this parameters will be increase by 1
      random_motion_force: float
          Magnitude impact of the random motion.
          If it is 0.0, then the points will not be moved
      """
  N,D = points.shape
  # For each dimension
  for d in range(D):
      # Get the minimum and maximum values for each point in this dimension
      min_d = np.min(points[:,d])
      max_d = np.max(points[:,d])

      # Estimate a small value based in the total of points and
      # points position in the dimension
      range_uniform = (max_d - min_d)/N

      # Apply a random motion in the points with the estimated motion value
      # and the force specified by user 
      points[:,d]=points[:,d] + random_motion_force*np.random.uniform(-range_uniform,range_uniform,N)
      

def proc_and_vis(
  dataX,
  dataY,
  dim_red_type='select_kbest_chi',
  dim_red_size=16,
  norm_type='stdscaller',
  vis_type='tsne',
  fig_name=None
):
  if norm_type == 'stdscaller':
    dataX = preprocessing.StandardScaler().fit_transform(dataX)
    mind = np.min(dataX)
     
  if dim_red_type == 'pca':
    dataX = PCA(n_components=dim_red_size).fit_transform(dataX)
  if dim_red_type == 'select_kbest_chi':
    if mind < 0:
      dataX += abs(mind)
    dataX = SelectKBest(chi2, k=dim_red_size).fit_transform(dataX, dataY)
  
  # dataX = dataX[:10000]
  # dataY = dataY[:10000]

  random_idx = np.random.choice(dataX.shape[0], 10000, replace=False)
  dataX = dataX[random_idx, :]
  dataY = dataY[random_idx]

  if vis_type == 'tsne':
    random_motion(dataX, 1.0)
    emb = TSNE(n_components=2, verbose=2, n_jobs=-1).fit_transform(dataX)
    # emb = TSNE(n_components=2, num_neighbors=32, verbose=2, warpwidth=4).fit_transform(dataX)
  if vis_type == 'pca':
    random_motion(dataX, 1.0)
    emb = PCA(n_components=2).fit_transform(dataX)
    # emb = TSNE(n_components=2, num_neighbors=32, verbose=2, warpwidth=4).fit_transform(dataX)
  

  # plt.scatter(x, y, c=label, cmap=matplotlib.colors.ListedColormap(colors))
  # plt.scatter(emb[:,0], emb[:,1], c=dataY, n_jobs=-1)
  
  fig = plt.figure(figsize=(8*2,6*2))
  if emb.shape[1] == 3:
      ax = fig.add_subplot(111, projection='3d')
  else:
      ax = fig.add_subplot(111)
  
  y = 'normal'
  ax.scatter(emb[dataY==y, 0], emb[dataY==y, 1], label=y, s=10, alpha=0.1)

  for y in set(dataY) - {'normal'}:
    ax.scatter(emb[dataY==y, 0], emb[dataY==y, 1], label=y, s=10, alpha=0.1)
  
  # lgnd = ax.legend()
  # lgnd = ax.legend()
  # lgnd = ax.legend(bbox_to_anchor=(1.0, 0.75, 0.0, 0.0))
  # lgnd = ax.legend(bbox_to_anchor=(1.0, 0.75, 0.0, 0.0),  prop={'size': 24})
  lgnd = ax.legend(prop={'size': 24})
  #change the marker size manually for both lines
  for y in lgnd.legendHandles:
      y._sizes = [80]
      # y._alphas = [1.0]
      y.set_alpha(1)



  ax = plt.gca()
  ax.xaxis.set_ticklabels([])
  ax.yaxis.set_ticklabels([])
  if emb.shape[1] == 3:
    ax.zaxis.set_ticklabels([])

  for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
  for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
  if emb.shape[1] == 3:
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)

  # plt.legend()
  if fig_name is None:
    plt.show()
  else:
    fig.tight_layout()
    fig.savefig(fig_name)


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  parser.add_argument('-o', '--output_file', dest='output_file', type=str,
                      required=False, default='result_toniot.json')
  parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
                      type=int)
  parser.add_argument('-s', dest='silent', action='store_true')

  parser.set_defaults(list=[])    
  parser.set_defaults(silent=False)
  
  args = parser.parse_args()
  print(args.input_file, args.list, args.silent, args.output_file)

  arg_test = [
    # {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    
    # {'use_dimred':True, 'class_red_type': 'etc_select', 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred':True, 'class_red_type': 'rbf', 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'use_dimred':True, 'class_red_type': 'pca', 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_type':'multiclass', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts']},
    # {'class_type':'multiclass', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label']},
    # {'class_type':'binary', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_type':'binary', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts']},
    # {'class_type':'binary', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label']},



    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'etc_select', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'pca', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    # {'class_red_type': 'rbf', 'use_dimred': True, 'class_type':'one_attack', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_ip', 'src_port', 'dst_ip', 'dst_port']},
    
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
    
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'normal', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    # {'use_dimred': False, 'class_type':'one_attack_three_classes', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    
    # {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'normal', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'scanning', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'ddos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'injection', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'mitm', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'dos', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'ransomware', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'xss', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'password', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    {'use_dimred': False, 'class_type':'one_attack_against_all', 'attack_type': 'backdoor', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']},
    
  ]
  
  


  result_data = []
  trained_avg_models = []

  for ton_args in arg_test:
  
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

    trainX, trainY, testX, testY, feature_names, ds_name = load_toniot(args.input_file, **ton_args_copy)
    print(ds_name)

    
    total_data_X = np.concatenate([trainX, testX])
    total_data_Y = np.concatenate([trainY, testY])
    
    le = preprocessing.LabelEncoder()
    le.fit(total_data_Y)

    set_y = set(total_data_Y)

    src_ip_idx = feature_names.index('src_ip')
    dst_ip_idx = feature_names.index('dst_ip')

    ip_set = set(total_data_X[:, src_ip_idx]).union(set(total_data_X[:, dst_ip_idx]))
    ip_set = list(ip_set)

    total_ips = len(ip_set)

    fl_dataX = {}
    fl_dataY = {}

    mask_total_data_X_local_ips = np.zeros(total_data_X.shape[0], dtype=np.bool)

    for ip in ip_set:
      if is_ip_private(ip):
        mask = total_data_X[:, src_ip_idx] == ip
        mask = np.logical_or(mask, total_data_X[:, dst_ip_idx] == ip)

        mask_total_data_X_local_ips = np.logical_or(mask_total_data_X_local_ips, mask)

        dataX = total_data_X[mask]

        dataY = total_data_Y[mask]
        fl_dataX[ip] = dataX
        fl_dataY[ip] = dataY
      # else:
        
      #   mask = total_data_X[:, src_ip_idx] == ip
      #   mask = np.logical_or(mask, total_data_X[:, dst_ip_idx] == ip)

      #   mask_total_data_X_local_ips = np.logical_or(mask_total_data_X_local_ips, mask)

      #   dataX = total_data_X[mask]

      #   dataY = total_data_Y[mask]
      #   ip = 'Internet'
      #   if  'Internet' in fl_dataX:
      #     print(ip)
      #     fl_dataX[ip].append(fl_dataX[ip])
      #     fl_dataY[ip].append(fl_dataY[ip])
      #   else:
      #     fl_dataX[ip] = [dataX]
      #     fl_dataY[ip] = [dataY]
                    
      # else:
      #   mask = total_data_X[:, src_ip_idx] == ip
      #   mask = np.logical_or(mask, total_data_X[:, dst_ip_idx] == ip)

      #   mask_total_data_X_local_ips = np.logical_or(mask_total_data_X_local_ips, mask)
      #   # dataX = total_data_X[mask]
      #   dataY = total_data_Y[mask]
      #   print(set(dataY))
    
    print("a")
    fl_dataX['internet'] = []
    fl_dataY['internet'] = []
    for x,y in zip(total_data_X, total_data_Y):
      if(not is_ip_private(x[src_ip_idx]) or (not is_ip_private(x[dst_ip_idx]))):
        fl_dataX['internet'].append(x)
        fl_dataY['internet'].append(y)
    
    fl_dataX['internet'] = np.array(fl_dataX['internet'])
    fl_dataY['internet'] = np.array(fl_dataY['internet'])
    # mask = ~np.isin(total_data_X[:, src_ip_idx], ip_set)
    # mask = np.logical_or(mask, ~np.isin(total_data_X[:, dst_ip_idx], ip_set) == ip)
    # dataX = total_data_X[mask]
    # dataY = total_data_Y[mask]
    # fl_dataX['internet'] = dataX
    # fl_dataY['internet'] = dataY
    print("b")

    fl_data_keys = [x for x in fl_dataX]
    
    # fl_data_keys = fl_data_keys+['127.0.0.1']
    # mask = np.logical_and(np.logical_not(np.isin(total_data_X[:, src_ip_idx], fl_data_keys)), np.logical_not(np.isin(total_data_X[:, dst_ip_idx], fl_data_keys)))
    # print(set(total_data_Y[mask]))
    # print(total_data_X[mask])
    # exit()
    total_dataY = [fl_dataY[x] for x in fl_data_keys]
    total_dataX = [fl_dataX[x] for x in fl_data_keys]

    def create_model(input_shape, num_classes):
      model = tf.keras.models.Sequential()
      
      model.add(tf.keras.layers.Dense(100, activation='sigmoid', input_dim=input_shape))
      # model.add(tf.keras.layers.Dense(1000, activation='sigmoid', input_dim=input_shape))
      model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
      
      opt = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
      # opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
      # opt = tf.keras.optimizers.Adam()
      model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
      return model
    

    num_classes = len(set_y)

    input_shape = fl_dataX[list(fl_dataX)[0]].shape[1] - 2

    models = {x: create_model(input_shape, num_classes) for x in fl_dataX}
    
    avg_model = create_model(input_shape, num_classes)

    # for round_id in range(100):
    # for round_id in range(10):
    for round_id in range(3):
    # for round_id in range(0):
      model_to_be_avg = {}
      for x in models:
        models[x].set_weights(avg_model.get_weights())
        model = models[x]
        dataX = fl_dataX[x]
        np.random.shuffle(dataX)

        dataX = np.delete(dataX, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)

        # count_dict = {y:sum(fl_dataY[x]==y) for y in set(fl_dataY[x])}
        # print(count_dict)
        # continue

        if len(set(fl_dataY[x])) <= 1:
          continue
        
        t_init = time.time()
        other_id = np.where(fl_dataY[x] == 'other')[0]
        attack_id = np.where(fl_dataY[x] != 'other')[0]
        # to_drop = np.random.choice(len(other_id), len(other_id) - min(len(other_id), len(attack_id)), replace=False)
        to_drop = np.random.choice(len(other_id), max(0,len(other_id) - 100), replace=False)
        # print(set(fl_dataY[x]), len(to_drop), len(other_id), len(attack_id), len(fl_dataY[x]))
        to_drop = other_id[to_drop]
        
        to_drop2 = np.random.choice(len(attack_id), max(0, len(attack_id) - 100), replace=False)
        to_drop2 = attack_id[to_drop2]

        to_drop = np.concatenate((to_drop, to_drop2))

        # dataY = np.array(fl_dataY[x])
        dataY = le.transform(fl_dataY[x]).astype(np.int32)

        dataX = np.delete(dataX, to_drop, 0)
        dataY = np.delete(dataY, to_drop, 0)
        # print(time.time() - t_init)

        # count_dict = {y:sum(dataY==y) for y in set(dataY)}
        # print(count_dict)

        # dataY[dataY == 'other'] = 0
        # dataY[dataY == 'normal'] = 2
        # dataY[np.logical_and(dataY != 0, dataY != 2)] = 1

        dataY = np.array(dataY, dtype=np.int32)

        dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)
        # print(dataX.shape)
        # print(dataY.shape)

        # X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.33, random_state=42, stratify=y)


        
        model.fit(dataX, dataY, epochs=1, batch_size=64, verbose=0)
        # model.fit(dataX, dataY, epochs=1, batch_size=1, verbose=0)
        # model.fit(dataX, dataY, epochs=1, batch_size=16, verbose=0)
        # model.fit(dataX, dataY, epochs=5, batch_size=1, verbose=0)
        results = model.evaluate(dataX, dataY, batch_size=1, verbose=0)
        train_size = len(dataY)
        print(f'{x}: {results[0]}, {results[1]}, {train_size}')

        model_to_be_avg[x] = model

      # total_train_samples = sum([len(fl_dataX[x]) for x in fl_dataX])
      # avg_contribution = {x: float(len(fl_dataX[x]))/total_train_samples for x in fl_dataX}
      avg_weights = sum([np.array(models[x].get_weights()) for x in models])/len(models)
      # avg_weights = sum([np.array(models[x].get_weights())*avg_contribution[x] for x in models])/len(models)

      
      # total_train_samples = sum([len(fl_dataX[x]) for x in model_to_be_avg])
      # avg_contribution = {x: float(len(fl_dataX[x]))/total_train_samples for x in model_to_be_avg}
      # avg_weights = sum([np.array(model_to_be_avg[x].get_weights())*avg_contribution[x] for x in model_to_be_avg])/len(model_to_be_avg)
      
      avg_model.set_weights(avg_weights)

      dataX = np.delete(total_data_X, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
      
      # dataY = np.array(total_data_Y)
      dataY = le.transform(total_data_Y).astype(np.int32)

      # dataY[dataY == 'other'] = 0
      # dataY[dataY == 'normal'] = 2
      # dataY[np.logical_and(dataY != 0, dataY != 2)] = 1
      
      dataY = np.array(dataY, dtype=np.int32)
        
      dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)
      dataX = dataX[mask_total_data_X_local_ips]
      dataY = dataY[mask_total_data_X_local_ips]

      results = avg_model.evaluate(dataX, dataY, batch_size=64, verbose=0)
      # p = np.array(avg_model.predict(dataX, batch_size=64, verbose=0))
      # print(p[:5])
      # print(dataY[:5])
      # print(total_data_Y[:5])
      # exit()
      # print(p)
      # print(type(p))
      # print(p.shape)
      # exit()

      loss = results[0]
      acc = results[1]
      print(f"Avg acc: {acc}")
      print(f"Avg loss: {loss}")

    # exit()
    trained_avg_models.append(avg_model)

    








  ##############################################################################

  ton_args = {'use_dimred':False, 'class_type':'multiclass', 'attack_type': 'all', 'return_fnames': True, 'test_ratio': 0.33 , 'drop_cols': ['type', 'label', 'ts', 'src_port', 'dst_port']}

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

  trainX, trainY, testX, testY, feature_names, ds_name = load_toniot(args.input_file, **ton_args_copy)
  print(ds_name)

  
  total_data_X = np.concatenate([trainX, testX])
  total_data_Y = np.concatenate([trainY, testY])
  
  le = preprocessing.LabelEncoder()
  le.fit(total_data_Y)

  set_y = set(total_data_Y)

  src_ip_idx = feature_names.index('src_ip')
  dst_ip_idx = feature_names.index('dst_ip')

  ip_set = set(total_data_X[:, src_ip_idx]).union(set(total_data_X[:, dst_ip_idx]))
  ip_set = list(ip_set)

  total_ips = len(ip_set)

  fl_dataX = {}
  fl_dataY = {}

  

  mask_total_data_X_local_ips = np.zeros(total_data_X.shape[0], dtype=np.bool)
  for ip in ip_set:
    if is_ip_private(ip):
      mask = total_data_X[:, src_ip_idx] == ip
      mask = np.logical_or(mask, total_data_X[:, dst_ip_idx] == ip)

      mask_total_data_X_local_ips = np.logical_or(mask_total_data_X_local_ips, mask)

      dataX = total_data_X[mask]

      dataY = total_data_Y[mask]
      fl_dataX[ip] = dataX
      fl_dataY[ip] = dataY
  
  # fl_data_keys = [x for x in fl_dataX]
  # total_dataY = [fl_dataY[x] for x in fl_data_keys]
  # total_dataX = [fl_dataX[x] for x in fl_data_keys]
  # TODO Concatenar fl_dataX para total_data usando local ips

  '''
  for x in models:
    models[x].set_weights(avg_model.get_weights())
    model = models[x]
    dataX = fl_dataX[x]
    np.random.shuffle(dataX)

    dataX = np.delete(dataX, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)

    dataY = le.transform(fl_dataY[x]).astype(np.int32)
    dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)

    model.fit(dataX, dataY, epochs=1, batch_size=64, verbose=0)

  total_train_samples = sum([len(fl_dataX[x]) for x in fl_dataX])
  avg_contribution = {x: float(len(fl_dataX[x]))/total_train_samples for x in fl_dataX}

  # avg_weights = sum([np.array(models[x].get_weights()) for x in models])/len(models)
  avg_weights = sum([np.array(models[x].get_weights())*avg_contribution[x] for x in models])/len(models)
  avg_model.set_weights(avg_weights)
  '''

  dataX = np.delete(total_data_X, np.s_[[src_ip_idx, dst_ip_idx]],axis=1).astype(np.float32)
  
  dataY = le.transform(total_data_Y).astype(np.int32)

  dataY = np.array(total_data_Y)
  '''
  dataY[dataY == 'other'] = 0
  dataY[dataY == 'normal'] = 2
  dataY[np.logical_or(dataY != 0, dataY != 2)] = 1
  dataY = np.array(dataY, dtype=np.int32)
        
  dataY = tf.keras.utils.to_categorical(dataY, num_classes = num_classes)
  '''
  dataX = dataX[mask_total_data_X_local_ips]
  dataY = dataY[mask_total_data_X_local_ips]

  predictions_per_model = []

  predictions = []
  
  # '''
  for model in trained_avg_models:
    # results = avg_model.evaluate(dataX, dataY, batch_size=64, verbose=0)
    # loss = results[0]
    # acc = results[1]
    # print(f"Avg acc: {acc}")
    p = np.array(model.predict(dataX, batch_size=64, verbose=0))
    predictions.append(p[:,1])
  
  predictions_proba = np.array(predictions).T
  attack_types = [ton_args['attack_type'] for ton_args in arg_test]
  predictions_best = [x[np.argmax(x)] for x in predictions_proba]
  predictions = [attack_types[np.argmax(x)] for x in predictions_proba]
  # '''


  '''
  for model in trained_avg_models:
    p = np.array(model.predict(dataX, batch_size=64, verbose=0))
    # predictions.append(p[:,1])
    predictions_per_model.append(p)
  
  predictions_per_model = np.array(predictions_per_model)

  
  attack_types = [ton_args['attack_type'] for ton_args in arg_test]
  count_normal = 0
  count_att = 0
  for i in range(predictions_per_model.shape[1]):
    _p = predictions_per_model[:, i, :]
    mask = np.argmax(_p, axis=1) == 1
    if np.sum(mask) == 0:
      predictions.append('normal')
      count_normal+=1
    else:
      count_att+=1
      pred_class_id = -1
      best_prob = -1
      for j, probs_model in enumerate(_p):
        if (np.argmax(probs_model) == 1) and probs_model[1] > best_prob:
          pred_class_id = j
          best_prob = probs_model[1]

      predictions.append(attack_types[pred_class_id])

  print(_p)

  print(f'count_normal: {count_normal}, count_att: {count_att}')
  '''



  comparison_matrix = np.array([predictions, dataY]).T
  '''
  # class_pred = np.argmax(predictions, axis=1)
  predictions_proba = np.array(predictions).T
  attack_types = [ton_args['attack_type'] for ton_args in arg_test]
  predictions_best = [x[np.argmax(x)] for x in predictions_proba]
  predictions = [attack_types[np.argmax(x)] for x in predictions_proba]
  comparison_matrix = np.array([predictions, dataY, predictions_best]).T
  '''

  print(comparison_matrix)
  print(sum(comparison_matrix[:,0] == comparison_matrix[:,1])/ len(comparison_matrix))

  for at in attack_types:
    p = sum(dataY == at)/ len(dataY)
    print(f'{at}: {p}')

  

if __name__ == "__main__":
  main()