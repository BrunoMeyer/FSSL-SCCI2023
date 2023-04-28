import pandas as pd
import numpy as np
import argparse


from cybersecurity_datasets import load_nslkdd, load_toniot, load_botiot

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

 
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


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
    y_pred = ssl_fl_solution.predict(self.testX)
    report = classification_report(self.testY, y_pred)

    print(report)

  def report_metrics_on_cross_val(self, solution, n_folds=10):

    total_dataX = np.concatenate((self.trainX, self.testX))
    total_dataY = np.concatenate((self.trainY, self.testY))

    skf = StratifiedKFold(n_splits=n_folds)

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
    dirichlet_beta = 100
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


    client_dataX, server_dataX, client_dataY, server_dataY = train_test_split(
    trainX, trainY, test_size=0.0009, random_state=42,
    stratify=trainY,
    shuffle=True
    )


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

    
    fl_dataX = [trainX[d[4][i]] for i in d[4]]
    fl_dataY = [trainY[d[4][i]] for i in d[4]]


    SSLFLProblem.__init__(self, fl_dataX, server_dataX, server_dataY, testX, testY, clients_dataY=fl_dataY)
    return


    
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
      # n_rounds = 10
      # n_rounds = 3
      # n_rounds = 1
      n_rounds = 0
      
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

  

if __name__ == "__main__":
  main()