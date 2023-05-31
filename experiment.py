import pandas as pd
import numpy as np
import argparse

from SSLFLCyberSecurityProblem import SSLFLCyberSecurityProblem
from SSLFLSimpleSSLFLSolution import SSLFLSimpleSSLFLSolution
from SSLFLPretextFLSolution import SSLFLPretextFLSolution

import pickle


def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  
  parser.add_argument('-en', '--exp_name', dest='exp_name', type=str,
                      required=False, default='test')

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


  # p = RandomGeneratedProblem()
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

  p = SSLFLCyberSecurityProblem(
    args.input_file,
    data_set_name=data_set_name,
    test_ratio=test_ratio,
    n_clients=n_clients,
    dirichlet_beta=dirichlet_beta,
    random_seed=random_seed)
  print("SSLFLCyberSecurityProblem")
  print("#"*80)


  # print("SimpleNonFLSolution")
  # s = SimpleNonFLSolution(p)
  # s.create()
  # # s.report_metrics()
  # s.plot_tsne_on_cross_val()


  # print("\n\SimpleFLSolution")
  # s = SimpleFLSolution(p)
  # s.create()
  # s.report_metrics()

  print("\n\SSLFLSimpleSSLFLSolution")
  s = SSLFLSimpleSSLFLSolution(p)
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

  print(log)

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
  with open('test.pickle', 'wb') as file_pi:
    pickle.dump(experiment_log, file_pi)
  

if __name__ == "__main__":
  main()