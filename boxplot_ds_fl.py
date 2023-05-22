# Import libraries
import matplotlib.pyplot as plt
import numpy as np
 
 
# Creating dataset
np.random.seed(0)
# data = np.random.normal(100, 20, 200)

toniot_clients = [46384, 44536, 42927, 44767, 45880, 47001, 47129, 45378, 46810, 45620]
botiot_clients = [309252, 296129, 276154, 285687, 285591, 282689, 303741, 297317, 294574, 303683]
nslkdd_clients = [46384, 44536, 42927, 44767, 45880, 47001, 47129, 45378, 46810, 45620]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(10, 3))
 
# Creating plot
ax1.boxplot(toniot_clients)
ax1.set_title("TonIoT")
ax2.boxplot(botiot_clients)
ax2.set_title("BotIoT")
ax3.boxplot(nslkdd_clients)
ax3.set_title("NSL-KDD")

fig.tight_layout(pad=3.0)
# show plot
plt.savefig("boxplot_ds_fl_dir100.pdf")
 
# Creating dataset
np.random.seed(0)
# data = np.random.normal(100, 20, 200)

toniot_clients = [17618, 18192, 310057, 4022, 13804, 7048, 26616, 18641, 38601, 1833]
botiot_clients = [11413, 17924, 1189166, 14160, 54, 60229, 49031, 130479, 1452748, 9613]
nslkdd_clients = [3533, 8112, 60928, 4869, 25911, 5362, 13509, 72, 1732, 1945]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize =(10, 3))
 
# Creating plot
ax1.boxplot(toniot_clients)
ax1.set_title("TonIoT")
ax2.boxplot(botiot_clients)
ax2.set_title("BotIoT")
ax3.boxplot(nslkdd_clients)
ax3.set_title("NSL-KDD")

fig.tight_layout(pad=3.0)
# show plot

fig.tight_layout(pad=3.0)
# show plot
plt.savefig("boxplot_ds_fl_dir0_1.pdf")
 