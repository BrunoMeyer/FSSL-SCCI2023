import pandas as pd
import numpy as np
import argparse

from SSLFLCyberSecurityProblem import SSLFLCyberSecurityProblem
from SSLFLSimpleSSLFLSolution import SSLFLSimpleSSLFLSolution
from SSLFLPretextFLSolution import SSLFLPretextFLSolution

import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  # parser.add_argument('-f', '--input_file', dest='input_file', type=str,
  #                     required=True)
  
  parser.add_argument('-f','--input_file', nargs='+', dest='input_file', help='List of input files',
                      type=str, required=True)


  parser.add_argument('-a', '--analysis', dest='analysis', type=str,
                      default='loss')

  # parser.add_argument('-o', '--output_file', dest='output_file', type=str,
  #                     required=False, default=None)
  
  # parser.add_argument('-en', '--exp_name', dest='exp_name', type=str,
  #                     required=False, default='test')

  # parser.add_argument('-d', '--data_set_name', dest='data_set_name', type=str,
  #                     required=False, choices=['toniot', 'botiot', 'nsl-kdd'], default='toniot')

  # parser.add_argument('-rs', '--random_seed', dest='random_seed', type=int,
  #                     required=False, default=0)

  # parser.add_argument('-nc', '--n_clients', dest='n_clients', type=int,
  #                     required=False, default=10)

  # parser.add_argument('-tr', '--test_ratio', dest='test_ratio', type=float,
  #                     required=False, default=0.01)

  # parser.add_argument('-db', '--dirichlet_beta', dest='dirichlet_beta', type=float,
  #                     required=False, default=100.0)

  # parser.add_argument('-nr', '--n_rounds', dest='n_rounds', type=int,
  #                     required=False, default=10)

  # parser.add_argument('-nes', '--n_epochs_server', dest='n_epochs_server', type=int,
  #                     required=False, default=300)

  # parser.add_argument('-nec', '--n_epochs_client', dest='n_epochs_client', type=int,
  #                     required=False, default=1)

  # parser.add_argument('-vs', '--verbose_server', dest='verbose_server', type=int,
  #                     required=False, default=1)

  # parser.add_argument('-vc', '--verbose_client', dest='verbose_client', type=int,
  #                     required=False, default=0)


  parser.set_defaults(list=[])

  args = parser.parse_args()



  # experiment_log = {
  #   'meta': {
  #     'version': 1,
  #     'n_rounds': args.n_rounds,
  #     'name': args.exp_name,
  #     'epochs_client': args.n_epochs_client,
  #     'epochs_server': args.n_epochs_server,
  #     'verbose_client': args.verbose_client,
  #     'verbose_server': args.verbose_server,

  #     'data_set_name': data_set_name,
  #     'test_ratio': test_ratio,
  #     'n_clients': n_clients,
  #     'dirichlet_beta': dirichlet_beta,
  #     'random_seed': random_seed
  #   },
  #   'data': log
  # }

  if args.analysis == 'loss':
    if len(args.input_file) == 1:
      input_file = args.input_file[0]
      with open(input_file, "rb") as file_pi:
        log = pickle.load(file_pi)
      
      for fold in log['data']['log_create_folds']:
        client_y_list = {}
        for round_client in fold['log_rounds_autoencoder']:
          for i, client in enumerate(round_client):
            if i in client_y_list:
              client_y_list[i].append(client['loss'][0])
            else:
              client_y_list[i] = [client['loss'][0]]
            
        print(client_y_list)

        fig, ax = plt.subplots(1, figsize=(8,8))
        for i in client_y_list:
          client_losses = client_y_list[i]
          ax.plot(np.arange(len(client_y_list[0])), client_losses, label=f'Client {i}')
        
        plt.legend()
        plt.show()

  if args.analysis == 'loss_final_model':
    if len(args.input_file) == 1:
      input_file = args.input_file[0]
      with open(input_file, "rb") as file_pi:
        log = pickle.load(file_pi)
      
      fold_y_list = {}
      for i, fold in enumerate(log['data']['log_create_folds']):
        hist_loss = fold['log_final_model']['loss']
        fold_y_list[i] = hist_loss
            
        print(fold_y_list)

      fig, ax = plt.subplots(1, figsize=(8,8))
      for i in fold_y_list:
        hist_losses = fold_y_list[i]
        ax.plot(np.arange(len(hist_losses)), hist_losses, label=f'Fold {i}')
      
      plt.legend()
      plt.show()


  if args.analysis == 'clients':
    input_file = args.input_file[0]

    n_clients_list = []
    fs_list = []
    for input_file in args.input_file:
      with open(input_file, "rb") as file_pi:
        log = pickle.load(file_pi)
      
      n_clients_list.append(log['meta']['n_clients'])
      if log['meta']['n_rounds'] == 0:
        n_clients_list[-1] = 0
      fs_client_list = []
      for fold in log['data']['log_pred_folds']:
        # print(sum(fold['predY'] == fold['testY'])/len(fold['predY']))
        # fs = f1_score(fold['testY'], fold['predY'], average='macro')
        fs = f1_score(fold['testY'], fold['predY'], average='weighted')
        fs_client_list.append(fs)

      fs_list.append(np.mean(fs_client_list))
      

        # print(client_y_list)

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.plot(n_clients_list, fs_list)
    
    plt.show()


  
  

if __name__ == "__main__":
  main()