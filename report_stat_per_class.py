#!/usr/bin/env python3

import pandas as pd
import numpy as np
import argparse

def main():
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('-f', '--input_file', dest='input_file', type=str,
                      required=True)
#   parser.add_argument('-l','--list', nargs='+', dest='list', help='List of ',
#                       type=int)
#   parser.add_argument('-s', dest='silent', action='store_true')

#   parser.set_defaults(list=[])    
#   parser.set_defaults(silent=False)
  
  args = parser.parse_args()
  print(args.input_file)

  convert_ds_class = None
  if 'toniot' in args.input_file:
    convert_ds_class = {0: 'backdoor', 1: 'ddos', 2: 'dos', 3: 'injection', 4: 'mitm', 5: 'normal', 6: 'password', 7: 'ransomware', 8: 'scanning', 9: 'xss'}
  if 'botiot' in args.input_file:
    convert_ds_class = {0: 'DDoS', 1: 'DoS', 2: 'Normal', 3: 'Reconnaissance'}
  if 'nslkdd' in args.input_file:
    convert_ds_class = {0: 'apache2', 1: 'back', 2: 'buffer_overflow', 3: 'ftp_write', 4: 'guess_passwd', 5: 'httptunnel', 6: 'imap', 7: 'ipsweep', 8: 'land', 9: 'loadmodule', 10: 'mailbomb', 11: 'mscan', 12: 'multihop', 13: 'named', 14: 'neptune', 15: 'nmap', 16: 'normal', 17: 'perl', 18: 'phf', 19: 'pod', 20: 'portsweep', 21: 'processtable', 22: 'ps', 23: 'rootkit', 24: 'saint', 25: 'satan', 26: 'sendmail', 27: 'smurf', 28: 'snmpgetattack', 29: 'snmpguess', 30: 'spy', 31: 'teardrop', 32: 'warezclient', 33: 'warezmaster', 34: 'xlock', 35: 'xsnoop', 36: 'xterm'}

  with open(args.input_file, 'r') as f:
    data = f.readlines()
  
  classes_f1_score = {}
  for line in data:
    if (line.startswith('       ')) and (not ('support' in line)):
      line = line.replace('\n', '')
      # cols = line.split('       ')
      cols = line.split()
      # print(len(cols) != 5)
      # metrics = cols[2]
      # print(cols[0], cols[3])
      # print(metrics)
      c = cols[0]
      f1s = float(cols[3])
      if c in classes_f1_score: 
        classes_f1_score[c].append(f1s)
      else:
        classes_f1_score[c] = [f1s]
  
  for c in sorted(list(classes_f1_score.keys())):
    real_c = convert_ds_class[int(c)]
    print('{:<8}\t{:.3f}'.format(real_c, np.mean(classes_f1_score[c])))

if __name__ == "__main__":
  main()