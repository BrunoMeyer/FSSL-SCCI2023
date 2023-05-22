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

  p = SSLFLCyberSecurityProblem(args.input_file)
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
  s.report_metrics_on_cross_val()

  # s.plot_tsne_on_cross_val()
  # p.report_train_test_stats_cross_val()

  

if __name__ == "__main__":
  main()