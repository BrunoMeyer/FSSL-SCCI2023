import pandas as pd
import numpy as np
import argparse

from SSLFLCyberSecurityProblem import SSLFLCyberSecurityProblem
from SSLFLSimpleSSLFLSolution import SSLFLSimpleSSLFLSolution

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
  

  args = parser.parse_args()


  # p = RandomGeneratedProblem()
  # print("RandomGeneratedProblem")
  # print("#"*80)

  # p = SSLFLCyberSecurityProblem(args.input_file)

  data_set_name='botiot'
  # data_set_name='nsl-kdd'
  test_ratio = 0.01
  n_clients = 10
  # n_clients = 50
  # dirichlet_beta=0.1
  dirichlet_beta=100.0
  random_seed = 0

  p = SSLFLCyberSecurityProblem(args.input_file, data_set_name=data_set_name, test_ratio=test_ratio, n_clients=n_clients, dirichlet_beta=dirichlet_beta, random_seed=random_seed)
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
  # s.create()
  # s.report_metrics()
  s.report_metrics_on_cross_val(n_rounds=10, name='fssl')

  # s.plot_tsne_on_cross_val()
  # p.report_train_test_stats_cross_val()

  

if __name__ == "__main__":
  main()