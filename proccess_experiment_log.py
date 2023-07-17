#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

from SSLFLCyberSecurityProblem import SSLFLCyberSecurityProblem
from SSLFLSimpleSSLFLSolution import SSLFLSimpleSSLFLSolution
from SSLFLPretextFLSolution import SSLFLPretextFLSolution

import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

from scipy import spatial

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')

  # parser.add_argument('-f', '--input_file', dest='input_file', type=str,
  #                     required=True)
  
  parser.add_argument('-f','--input_file', nargs='+', dest='input_file', help='List of input files',
                      type=str, required=True)


  parser.add_argument('-a', '--analysis', dest='analysis', type=str,
                      default='loss',
                      choices=[
                        'clients_gradient',
                        'loss',
                        'clients',
                        'clients_eval_reconstruct_test',
                        'loss_final_model'])

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

  if args.analysis == 'clients_gradient':
    if len(args.input_file) == 1:
      input_file = args.input_file[0]
      with open(input_file, "rb") as file_pi:
        log = pickle.load(file_pi)
      
      fold_id = 0
      round_id = 0
      client_id = 0
      log_clients_dataY = log['data']['log_clients_dataY']

      # TODO: Cosine distance
      
      # for fold_id, fold_result in enumerate(log['data']['log_create_folds'][-1:]):
      for fold_id in [len(log['data']['log_create_folds'])-1]:
        fold_result = log['data']['log_create_folds'][fold_id]
        # for round_id, round_result in enumerate(fold_result['log_autoencoder_weights_init_rounds'][-1:]):
        # for round_id in [len(fold_result['log_autoencoder_weights_init_rounds'])-1]:
        for round_id in [0]:
          round_result = fold_result['log_autoencoder_weights_init_rounds'][round_id]
        # for round_id, round_result in enumerate(fold_result['log_autoencoder_weights_init_rounds'][:1]):
          init_final_model = fold_result['log_final_pretext_model_weights_init_rounds'][round_id]
          final_final_model = fold_result['log_final_pretext_model_weights_final_rounds'][round_id]

          total_diff = 0
          
          for client_id, init_weights in enumerate(round_result):
            final_weights = log['data']['log_create_folds'][fold_id]['log_autoencoder_weights_final_rounds'][round_id][client_id]
            # total_diff = []
            
            # for layer1, layer2, layer3, layer4 in zip(init_weights, final_weights, init_final_model, final_final_model):
            # for layer1, layer2, layer3, layer4 in zip(init_weights, final_weights, init_final_model, final_final_model):
            # for layerid in range(len(init_weights)):
            for layerid in range(2):
              # total_diff = np.concatenate((total_diff, layer1-layer2))
              layer1 = init_weights[layerid]
              layer2 = final_weights[layerid]
              layer3 = init_final_model[layerid]
              layer4 = final_final_model[layerid]

              gradient_client = layer2-layer1
              gradient_server = layer4-layer3

              # total_diff += np.sum(np.abs(gradient_server-gradient_client)) # L1 dist
              
              for gradient_neuron_c, gradient_neuron_s in zip(gradient_server, gradient_client):
                cd = spatial.distance.cosine(gradient_neuron_c, gradient_neuron_s) # Cos dist
                if isinstance(cd, np.floating) and not np.isnan(cd):
                  total_diff += np.abs(cd)

              # print(np.sum(np.abs(gradient_server-gradient_client))/len(log_clients_dataY[client_id]))
              # print(np.sum(np.abs(gradient_server-gradient_client))/2)
              print(np.sum(np.abs(gradient_server-gradient_client)))
            # print(total_diff)
            print("")
          print("#"*30)
          print("total_diff: {}".format(total_diff/len(round_result)))
          print("#"*30)

      exit()
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
        # if fs > 0.1:
        #   fs_client_list.append(fs)
        fs_client_list.append(fs)

      # fs_list.append(np.mean(fs_client_list))
      fs_list.append(fs_client_list)
      

        # print(client_y_list)

    fig, ax = plt.subplots(1, figsize=(8,8))

    print(list(zip(n_clients_list, fs_list)))
    # ax.plot(n_clients_list, fs_list)

    ax.boxplot(fs_list)
    ax.set_xticklabels(n_clients_list)
    # ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    
    plt.show()

  if args.analysis == 'clients_eval_reconstruct_test':

    if len(args.input_file) == 1:
      input_file = args.input_file[0]
      n_clients_list = []
      fs_list = []
      error = []
      with open(input_file, "rb") as file_pi:
        log = pickle.load(file_pi)

      fig, ax = plt.subplots(1, figsize=(8,8))
      
      fs_client_list = []
      for fold in log['data']['log_create_folds']:
        # print(sum(fold['predY'] == fold['testY'])/len(fold['predY']))
        # fs = f1_score(fold['testY'], fold['predY'], average='macro')
        # fs = f1_score(fold['testY'], fold['predY'], average='weighted')
        fs = [x[0] for x in fold['eval_reconstruct_test']]
        # if (sum(fs) > 600000):
        if True or (sum(fs) > 600000):
          fs_list.append(fs)
          ax.plot(np.arange(len(fs)), fs)
          ax.yaxis.set_major_locator(plt.MaxNLocator(15))

      # fig, ax = plt.subplots()
      # ax = fig.add_axes([0, 0, 1, 1])
      # ax.plot(n_clients_list, fs_list)
      # ax.errorbar(np.arange(len(fs_list[0])), fs_list, yerr=error, fmt='-o')
      # ax.boxplot(fs_list)
      # ax.set_xticklabels(n_clients_list)
    
    else:
      n_clients_list = []
      fs_list = []
      error = []
      for input_file in args.input_file:
        with open(input_file, "rb") as file_pi:
          log = pickle.load(file_pi)
        
        n_clients_list.append(log['meta']['n_clients'])
        if log['meta']['n_rounds'] == 0:
          n_clients_list[-1] = 0
        fs_client_list = []
        for fold in log['data']['log_create_folds']:
          # print(sum(fold['predY'] == fold['testY'])/len(fold['predY']))
          # fs = f1_score(fold['testY'], fold['predY'], average='macro')
          # fs = f1_score(fold['testY'], fold['predY'], average='weighted')
          # fs = fold['eval_reconstruct_test'][-1]
          fs = fold['eval_reconstruct_test'][0]
          if (not np.isnan(fs)) and (not (type(fs) is None)):
            # if (fs > 10000):
            if True or (fs > 10000):
              fs_client_list.append(fs)

        # fs_list.append(np.mean(fs_client_list))
        # error.append(np.std(fs_client_list))
        fs_list.append(fs_client_list)
        

          # print(client_y_list)

      fig, ax = plt.subplots(1, figsize=(8,8))
      # fig, ax = plt.subplots()
      # ax = fig.add_axes([0, 0, 1, 1])
      print(list(zip(n_clients_list, fs_list)))
      # ax.plot(n_clients_list, fs_list)
      # ax.errorbar(n_clients_list, fs_list, yerr=error, fmt='-o')
      ax.boxplot(fs_list)
      ax.set_xticklabels(n_clients_list)


    
    plt.show()


  
  

if __name__ == "__main__":
  main()
