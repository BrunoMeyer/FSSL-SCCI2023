#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')


import pandas as pd
import numpy as np
import argparse

from SSLFLProblem import RandomGeneratedProblem
from SSLFLCyberSecurityProblem import SSLFLCyberSecurityProblem
from SSLFLSimpleSSLFLSolution import SSLFLSimpleSSLFLSolution, SSLFLFreezeKTSSLFLSolution, SSLFLFreezeRepLearningKTSSLFLSolution
from SSLFLPretextFLSolution import SSLFLPretextFLSolution

import pickle


import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import os



def check_clients_quality(args, problem):
  assert not (problem.clients_dataY is None)
  n_clients = len(problem.clients_dataY)
  num_classes = problem.num_classes
  
  pairwise_client_result = np.zeros((n_clients, n_clients))
  for i in range(n_clients):
    print("Training client {}".format(i))
    model = SSLFLFreezeKTSSLFLSolution(problem).create_model_dl(problem.input_shape, num_classes)
    categ_clienttrainY = tf.keras.utils.to_categorical(problem.clients_dataY[i], num_classes = num_classes)
    # categ_trainY = tf.keras.utils.to_categorical(problem.trainY, num_classes = num_classes)
    # categ_testY = tf.keras.utils.to_categorical(problem.testY, num_classes = num_classes)
    
    model.fit(problem.clients_dataX[i], categ_clienttrainY, epochs=200, verbose=0)

    for j in range(n_clients):
      categ_testclienttrainY = tf.keras.utils.to_categorical(problem.clients_dataY[j], num_classes = num_classes)
      pairwise_client_result[i, j] = model.evaluate(problem.clients_dataX[j], categ_testclienttrainY)[1] # Accuracy

    # print("Train: :", model.evaluate(problem.trainX, categ_trainY))
    # print("Test: :", model.evaluate(problem.testX, categ_testY))
  
  ax = sns.heatmap(pairwise_client_result, linewidth=0.5)
  plt.savefig("{}_pairwise_client_result_heatmap.png".format(args.exp_name))

  # TODO: Identify better clients and run experiments with poor, medium and high quality clients
  with open('pairwise_client_result_{}.pickle'.format(args.exp_name), 'wb') as f:
    pickle.dump(pairwise_client_result, f)

  return pairwise_client_result
  

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  
  parser.add_argument('-o', '--output_file', dest='output_file', type=str,
                      required=False, default=None)
  
  parser.add_argument('-en', '--exp_name', dest='exp_name', type=str,
                      required=False, default='test')
  
  parser.add_argument('-cp', '--clients_preference', dest='clients_preference', type=str,
                      required=False, default=None, choices=['best', 'worse'])
  
  parser.add_argument('-pc', '--preference_count', dest='preference_count', type=int,
                      required=False, default=None)

  parser.add_argument('-d', '--data_set_name', dest='data_set_name', type=str,
                      required=False, choices=['toniot', 'botiot', 'nsl-kdd'], default='toniot')

  parser.add_argument('-rs', '--random_seed', dest='random_seed', type=int,
                      required=False, default=0)

  parser.add_argument('-nc', '--n_clients', dest='n_clients', type=int,
                      required=False, default=10)

  parser.add_argument('-tr', '--test_ratio', dest='test_ratio', type=float,
                      required=False, default=0.01)

  parser.add_argument('-db', '--dirichlet_beta', dest='dirichlet_beta', type=float,
                      required=False, default=100.0)

  parser.add_argument('-nr', '--n_rounds', dest='n_rounds', type=int,
                      required=False, default=10)

  parser.add_argument('-nes', '--n_epochs_server', dest='n_epochs_server', type=int,
                      required=False, default=300)

  parser.add_argument('-nec', '--n_epochs_client', dest='n_epochs_client', type=int,
                      required=False, default=1)

  parser.add_argument('-vs', '--verbose_server', dest='verbose_server', type=int,
                      required=False, default=1)

  parser.add_argument('-vc', '--verbose_client', dest='verbose_client', type=int,
                      required=False, default=0)



  args = parser.parse_args()


  # p = RandomGeneratedProblem(n_clients=args.n_clients)
  # print("RandomGeneratedProblem")
  # print("#"*80)

  # p = SSLFLCyberSecurityProblem(args.input_file)

  # data_set_name='toniot'
  data_set_name = args.data_set_name
  # data_set_name='nsl-kdd'
  test_ratio = args.test_ratio
  n_clients = args.n_clients
  # n_clients = 50
  # dirichlet_beta=0.1
  dirichlet_beta = args.dirichlet_beta
  random_seed = args.random_seed
  output_file = args.output_file

  if output_file is None:
    clients_preference = "" if args.clients_preference is None else '-'+args.clients_preference

    output_file = '{}.pickle'.format(args.exp_name+clients_preference)


  p = SSLFLCyberSecurityProblem(
    args.input_file,
    data_set_name=data_set_name,
    test_ratio=test_ratio,
    n_clients=n_clients,
    dirichlet_beta=dirichlet_beta,
    random_seed=random_seed,
    normalize_data = True)


  print("SSLFLCyberSecurityProblem")
  print("#"*80)

  pairwise_client_quality = None
  if not (args.clients_preference is None):
    fpath = 'pairwise_client_result_{}.pickle'.format(args.exp_name)
    if os.path.exists(fpath):
      print('Loading data from ', fpath)
      with open(fpath, 'rb') as f:
        pairwise_client_quality = pickle.load(f)
    else:
      pairwise_client_quality = check_clients_quality(args, p)
  

  # print("SimpleNonFLSolution")
  # s = SimpleNonFLSolution(p)
  # s.create()
  # # s.report_metrics()
  # s.plot_tsne_on_cross_val()


  # print("\n\SimpleFLSolution")
  # s = SimpleFLSolution(p)
  # s.create()
  # s.report_metrics()

  # print("\n\SSLFLSimpleSSLFLSolution")
  # s = SSLFLSimpleSSLFLSolution(p)

  print("\n\SSLFLFreezeKTSSLFLSolution")
  s = SSLFLFreezeKTSSLFLSolution(p,
                                 pairwise_client_quality = pairwise_client_quality,
                                 clients_preference = args.clients_preference,
                                 preference_count = args.preference_count
                                 )

  # print("\n\SSLFLFreezeRepLearningKTSSLFLSolution")
  # s = SSLFLFreezeRepLearningKTSSLFLSolution(p,
  #                                pairwise_client_quality = pairwise_client_quality,
  #                                clients_preference = args.clients_preference,
  #                                preference_count = args.preference_count,
  #                                use_masked_autoencoder=False
  #                                )
  
  # s = SSLFLPretextFLSolution(p)
  # s.create()
  # s.report_metrics()
  # s.report_metrics_on_cross_val(n_rounds=100, name='fssl')
  log = s.report_metrics_on_cross_val(
    n_rounds=args.n_rounds,
    name=args.exp_name,
    epochs_client=args.n_epochs_client,
    epochs_server=args.n_epochs_server,
    verbose_client=args.verbose_client,
    verbose_server=args.verbose_server
    )
  

  # s.report_metrics_on_cross_val(n_rounds=10, name='fssl')
  # s.report_metrics_on_cross_val(n_rounds=0, name='centralized')

  # s.plot_tsne_on_cross_val()
  # p.report_train_test_stats_cross_val()

  # print(log)

  
  experiment_log = {
    'meta': {
      'version': 1,
      'n_rounds': args.n_rounds,
      'name': args.exp_name,
      'epochs_client': args.n_epochs_client,
      'epochs_server': args.n_epochs_server,
      'verbose_client': args.verbose_client,
      'verbose_server': args.verbose_server,

      'data_set_name': data_set_name,
      'test_ratio': test_ratio,
      'n_clients': n_clients,
      'dirichlet_beta': dirichlet_beta,
      'random_seed': random_seed
    },
    'data': log
  }
  with open(output_file, 'wb') as file_pi:
    pickle.dump(experiment_log, file_pi)
  

if __name__ == "__main__":
  main()