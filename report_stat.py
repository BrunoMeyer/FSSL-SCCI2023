# Weight avg f1-score

import numpy as np
from scipy import stats

## TonIoT
print("TonIoT")
centr = [0.84, 0.86, 0.85, 0.86, 0.84, 0.86, 0.87, 0.87, 0.85, 0.83]
ffsl = [0.86, 0.88, 0.87, 0.89, 0.86, 0.85, 0.89, 0.89, 0.87, 0.87]
ffsl_ni = [0.85, 0.85, 0.85, 0.89, 0.87, 0.87, 0.89, 0.90, 0.87, 0.87]

print("Centr: ", np.mean(centr))
print("ffsl: ", np.mean(ffsl))
print("ffsl_ni: ", np.mean(ffsl_ni))
print("")

print("Centr - FFSL stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl)))
print("Centr - FFSL stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl)))
print("")

print("Centr - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl_ni)))
print("Centr - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl_ni)))
print("")

print("FFSL - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(ffsl, ffsl_ni)))
print("FFSL - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(ffsl, ffsl_ni)))


# BotIoT
print("")
print("")
print("BotIoT")

centr = [0.36, 0.28, 0.36, 0.29, 0.28, 0.36, 0.28, 0.28, 0.28, 0.28]
ffsl = [0.47, 0.44, 0.55, 0.60, 0.40, 0.49, 0.60, 0.53, 0.43, 0.57]
ffsl_ni = [0.40, 0.46, 0.42, 0.31, 0.39, 0.33, 0.36, 0.33, 0.42, 0.42]


print("Centr: ", np.mean(centr))
print("ffsl: ", np.mean(ffsl))
print("ffsl_ni: ", np.mean(ffsl_ni))
print("")

print("Centr - FFSL stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl)))
print("Centr - FFSL stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl)))
print("")

print("Centr - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl_ni)))
print("Centr - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl_ni)))
print("")

print("FFSL - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(ffsl, ffsl_ni)))
print("FFSL - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(ffsl, ffsl_ni)))


# BotIoT
print("")
print("")


# NSLKDD
print("NSLKDD")

centr = [0.90, 0.89, 0.88, 0.88, 0.87, 0.89, 0.86, 0.89, 0.88, 0.89]
ffsl = [0.89, 0.88, 0.90, 0.91, 0.87, 0.90, 0.87, 0.89, 0.89, 0.89]
ffsl_ni = [0.90, 0.90, 0.90, 0.91, 0.88, 0.90, 0.86, 0.88, 0.89, 0.90]

print("Centr: ", np.mean(centr))
print("ffsl: ", np.mean(ffsl))
print("ffsl_ni: ", np.mean(ffsl_ni))
print("")

print("Centr - FFSL stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl)))
print("Centr - FFSL stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl)))
print("")

print("Centr - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(centr, ffsl_ni)))
print("Centr - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(centr, ffsl_ni)))
print("")

print("FFSL - FFSL_NI stats.ttest_rel: {}".format(stats.ttest_rel(ffsl, ffsl_ni)))
print("FFSL - FFSL_NI stats.ttest_ind: {}".format(stats.ttest_ind(ffsl, ffsl_ni)))


# BotIoT
print("")
print("")
