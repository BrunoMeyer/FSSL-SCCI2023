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

    total_y = set()
    if not (clients_dataY is None):
      for x in clients_dataY:
        total_y = total_y.union(set(list(x)))
    
    if not (trainY is None):
      total_y = total_y.union(set(list(trainY)))
    if not (testY is None):
      total_y = total_y.union(set(list(testY)))

    self.num_classes = len(set(total_y))

  def report_metrics(self, ssl_fl_solution):
    y_pred = ssl_fl_solution.predict(self.testX)
    report = classification_report(self.testY, y_pred)

    print(report)

    return {'testY': self.testY, 'predY': y_pred}

  def plot_tsne(self, ssl_fl_solution, figname=None):
    latent_vectors_train = ssl_fl_solution.get_latent_space(self.trainX)
    latent_vectors_test = ssl_fl_solution.get_latent_space(self.testX)
    
    emb_test = TSNE(n_components=2).fit_transform(latent_vectors_test)
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    for y in set(list(self.testY)):
      y_realname = self.le.inverse_transform([y])[0]
      ax.scatter(emb_test[self.testY==y, 0], emb_test[self.testY==y, 1], label=y_realname, alpha=0.2)
    ax.legend()

    if figname is None:
      plt.show()
    else:
      fig.savefig('test_'+figname)
    
    emb_train = TSNE(n_components=2).fit_transform(latent_vectors_train)
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    for y in set(list(self.trainY)):
      y_realname = self.le.inverse_transform([y])[0]
      ax.scatter(emb_train[self.trainY==y, 0], emb_train[self.trainY==y, 1], label=y_realname, alpha=0.2)
    ax.legend()
    
    
    if figname is None:
      plt.show()
    else:
      fig.savefig('train_'+figname)
    


  def report_metrics_on_cross_val(self, solution, n_folds=10, **kargs):

    total_dataX = np.concatenate((self.trainX, self.testX))
    total_dataY = np.concatenate((self.trainY, self.testY))

    # skf = StratifiedKFold(n_splits=n_folds)
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)


    old_trainX = self.trainX
    old_trainY = self.trainY
    old_testX = self.testX
    old_testY = self.testY
    
    
    log_create_folds = []
    log_pred_folds = []

    for i, (test_index, train_index) in enumerate(skf.split(total_dataX, total_dataY)): # test index is bigger (inverted leave one out)
      self.trainX = total_dataX[train_index]
      self.trainY = total_dataY[train_index]
      self.testX = total_dataX[test_index]
      self.testY = total_dataY[test_index]

      print(f"Fold {i}")
      
      log_create = solution.create(**kargs)
      log_pred = solution.report_metrics()


      log_create_folds.append(log_create)
      log_pred_folds.append(log_pred)

    self.trainX = old_trainX
    self.trainY = old_trainY
    self.testX = old_testX
    self.testY = old_testY

    return {
      'log_create_folds': log_create_folds,
      'log_pred_folds': log_pred_folds,
    }


  def report_train_test_stats_cross_val(self, n_folds=10):

    total_dataX = np.concatenate((self.trainX, self.testX))
    total_dataY = np.concatenate((self.trainY, self.testY))

    # skf = StratifiedKFold(n_splits=n_folds)
    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)


    old_trainX = self.trainX
    old_trainY = self.trainY
    old_testX = self.testX
    old_testY = self.testY
    
    all_c = set(list(old_trainY)+list(old_testY))
    all_c = self.le.inverse_transform(list(all_c))

    dict_class_train_folds = {c:[] for c in all_c}
    dict_class_test_folds = {c:[] for c in all_c}
    
    total_train_folds = []
    total_test_folds = []
    for i, (test_index, train_index) in enumerate(skf.split(total_dataX, total_dataY)): # test index is bigger (inverted leave one out)
      self.trainX = total_dataX[train_index]
      self.trainY = total_dataY[train_index]
      self.testX = total_dataX[test_index]
      self.testY = total_dataY[test_index]

      total_train_folds.append(len(self.trainY))
      total_test_folds.append(len(self.testY))

      print(f"Fold {i}")
      trainY = self.le.inverse_transform(self.trainY)
      testY = self.le.inverse_transform(self.testY)
      dclass_train = collections.Counter(trainY)
      for c in dclass_train:
        dict_class_train_folds[c].append(dclass_train[c])

      dclass_test = collections.Counter(testY)      
      for c in dclass_test:
        dict_class_test_folds[c].append(dclass_test[c])
    
    print("Average train test: {}".format(np.mean(total_train_folds)))
    print("Average test test: {}".format(np.mean(total_test_folds)))


    print("Instances per class (train)")
    for c in all_c:
      print('{:<8}\t{:d}'.format(c, int(np.mean(dict_class_train_folds[c]))))

    print("Instances per class (test)")
    for c in all_c:
      print('{:<8}\t{:d}'.format(c, int(np.mean(dict_class_test_folds[c]))))


    self.trainX = old_trainX
    self.trainY = old_trainY
    self.testX = old_testX
    self.testY = old_testY

  
  def plot_tsne_on_cross_val(self, solution, n_folds=10):

    total_dataX = np.concatenate((self.trainX, self.testX))
    total_dataY = np.concatenate((self.trainY, self.testY))

    skf = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)


    old_trainX = self.trainX
    old_trainY = self.trainY
    old_testX = self.testX
    old_testY = self.testY
    
    
    for i, (test_index, train_index) in enumerate(skf.split(total_dataX, total_dataY)): # test index is bigger (inverted leave one out)
      self.trainX = total_dataX[train_index]
      self.trainY = total_dataY[train_index]
      self.testX = total_dataX[test_index]
      self.testY = total_dataY[test_index]

      print(f"Fold {i}")
      
      solution.create()
      self.plot_tsne(solution, figname=f"{self.data_set_name}_fold_{i}_{solution.name}.pdf")

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
      

      clients_dataX.append(X[test_index])
      clients_dataY.append(y[test_index])
    
    

    trainX = clients_dataX[0]
    trainY = clients_dataY[0]
    
    testX = clients_dataX[1]
    testY = clients_dataY[1]
    
    clients_dataX = clients_dataX[2:]
    clients_dataY = clients_dataY[2:]
    

    

    SSLFLProblem.__init__(self, clients_dataX, trainX, trainY, testX, testY, clients_dataY=clients_dataY)


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()

  

if __name__ == "__main__":
  main()