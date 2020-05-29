import theano
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl
from mnist import load_mnist
from convnet import init_weights, RMSprop, convnet_model
from associative import AssociativeMemory, AssociativeMemoryError
#%matplotlib inline

mnist_path = './mnist'

sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
######################################################################################

# Plot of precision and recall with entropies

print('Comienza ploteo-----')
precision = np.load('precision.npy')
recall = np.load('recall.npy')
entropies = np.load('entropies.npy')

average_entropy = entropies.mean(axis=1)
# Percentage
average_precision = precision[:, 10, :] * 100
average_recall = recall[:, 10, :] * 100

# Setting up a colormap that's a simple transtion
cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0,0],[0,0]]
step = 0.1
levels = np.arange(0.0, 90 + step, step)
CS3 = plt.contourf(Z, levels, cmap=cmap)

plt.clf()


plt.plot(np.arange(0, 100, 10), average_precision, 'r-o', label='Precision')
plt.plot(np.arange(0, 100, 10), average_recall, 'b-s', label='Recall')
plt.xlim(-0.1, 91)
plt.ylim(0, 102)
plt.xticks(np.arange(0, 100, 10), sizes)

plt.xlabel('Range Quantization Levels')
plt.ylabel('Percentage [%]')
plt.legend(loc=4)
plt.grid(True)

entropy_labels = [str(e) for e in np.around(average_entropy, decimals=1)]

cbar = plt.colorbar(CS3, orientation='horizontal')
cbar.set_ticks(np.arange(0, 100, 10))
cbar.ax.set_xticklabels(entropy_labels)
cbar.set_label('Entropy')

plt.savefig('graph_l4.png', dpi=500)
#plt.show()
