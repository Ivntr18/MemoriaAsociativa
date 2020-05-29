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

#################################################################
# Load data
trX, teX, trY, teY = load_mnist(mnist_path)
trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

X = T.ftensor4()
Y = T.fmatrix()

################################################################

# Setup of the training and testing.
# Do not run this cell if you already have the weights of the network. 

w1 = init_weights((32, 1, 3, 3))
w2 = init_weights((64, 32, 3, 3))
w3 = init_weights((128, 64, 3, 3))
w4 = init_weights((128 * 3 * 3, 625))
w5 = init_weights((625, 10))

# model with dropout ('n'oisy outputs)
n_l1, n_l2, n_l3, n_l4, n_py_x = convnet_model(X, w1, w2, w3, w4, w5, 0.2, 0.5)

# cost function
cost = T.mean(T.nnet.categorical_crossentropy(n_py_x, Y))
params = [w1, w2, w3, w4, w5]
updates = RMSprop(cost, params, lr=0.001)

# Train function
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)


# model without dropout
l1, l2, l3, l4, py_x = convnet_model(X, w1, w2, w3, w4, w5, 0., 0.)
y_x = T.argmax(py_x, axis=1)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# Train the network
for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print('Testing epoch number: {0}'.format(i))
    print(np.mean(np.argmax(teY, axis=1) == predict(teX)))

# Save the weights of the network


np.save('w1.npy', w1.get_value())
np.save('w2.npy', w2.get_value())
np.save('w3.npy', w3.get_value())
np.save('w4.npy', w4.get_value())
np.save('w5.npy', w5.get_value())


##################################################################################

# Load the network's parameters to generate the features
# Do not run this cell if you already generated the features. Skip to the next cell

# Shared variables
w1 = theano.shared(np.load('w1.npy'), name='w1')
w2 = theano.shared(np.load('w2.npy'), name='w2')
w3 = theano.shared(np.load('w3.npy'), name='w3')
w4 = theano.shared(np.load('w4.npy'), name='w4')
w5 = theano.shared(np.load('w5.npy'), name='w5')

# model
l1, l2, l3, l4, py_x = convnet_model(X, w1, w2, w3, w4, w5, 0., 0.)

generate = theano.function(inputs=[X], outputs=l4, allow_input_downcast=True)

# Generate features from the network 128*3*3->625

train_features = np.zeros((60000, (625)), theano.config.floatX)

for start, end in zip(range(0, len(trX), 200), range(200, (len(trX) + 1), 200)):
    print(start, end)
    batch = generate(trX[start:end])
    #print(len(batch[0]))
    train_features[start:end] = batch
    
np.save('train_features_l4.npy', train_features)

test_features = np.zeros((10000, (625)), theano.config.floatX)

for start, end in zip(range(0, len(teX), 200), range(200, (len(teX) + 1), 200)):
    print(start, end)
    batch = generate(teX[start:end])
    test_features[start:end] = batch
    
np.save('test_features_l4.npy', test_features)


# Load the features

train_X = np.load('train_features_l4.npy')
test_X = np.load('test_features_l4.npy')
trX, teX, trY, teY = load_mnist(mnist_path, onehot=False)
# The ranges of all the memories that will be trained
sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
# the domain size. The size of the output layer of the network
domain = 625
# Maximum value of the features in the train set
max_val = train_X.max()


# Train the different co-domain memories

tables = np.zeros((len(sizes), 10, 5), dtype=np.float64)
entropies = np.zeros((len(sizes), 10), dtype=np.float64)

def get_ams_results(i, s, domain, train_X, test_X, trY, teY):
    table = np.zeros((10, 5), dtype=np.float64)
    entropy = np.zeros((10, ), dtype=np.float64)
    ams = dict.fromkeys(range(10))
    #print(str(ams))
    for j in ams:
        # Create the memories with domain 's'
        ams[j] = AssociativeMemory(domain, s)
    # Round the values
    train_X_around = np.around(train_X * (s - 1) / max_val).astype(np.int16)
    test_X_around = np.around(test_X * (s - 1) / max_val).astype(np.int16)
    # Abstraction
    for x, y in zip(train_X_around, trY):
        ams[y].abstract(x, input_range=s)
    # Calculate entropies
    for j in ams:
        #print(j)
        entropy[j] = ams[j].entropy
    # Reduction
    for x, y in zip(test_X_around, teY):
        table[y, 0] += 1
        for k in ams:
            try:
                ams[k].reduce(x, input_range=s)
                if k == y:
                    table[y, 1] += 1
                else:
                    table[y, 2] += 1
                # confusion_mat[k, y] += 1
            except AssociativeMemoryError:
                if k != y:
                    table[y, 3] += 1
                else:
                    table[y, 4] += 1
    return (i, table, entropy)

list_tables_entropies = Parallel(n_jobs=3, verbose=50)(
    delayed(get_ams_results)(i, s, domain, train_X, test_X, trY, teY) for i, s in enumerate(sizes))

for i, table, entropy in list_tables_entropies:
    tables[i, :, :] = table
    entropies[i, :] = entropy

# Table columns
# 0.- Total count
# 1.- Able to reduce and it is the same number
# 2.- Able to reduce and it is not the same number
# 3.- Not able to reduce and it is not the same number
# 4.- Not able to reduce and it is the same number

##########################################################################################

# Calculate the precision and recall

print('Calculando precision y recall')
precision = np.zeros((len(sizes), 11, 1), dtype=np.float64)
recall = np.zeros((len(sizes), 11, 1), dtype=np.float64)

for i, s in enumerate(sizes):
    prec_aux = tables[i, :, 1] / (tables[i, :, 1] + tables[i, :, 2])
    recall_aux = tables[i, :, 1] / tables[i, :, 0]
    precision[i, 0:10, 0] = prec_aux[:]
    precision[i, 10, 0] = prec_aux.mean()
    recall[i, 0:10, 0] = recall_aux[:]
    recall[i, 10, 0] = recall_aux.mean()
    
#np.save('precision.npy', precision)
#np.save('recall.npy', recall)
#np.save('entropies.npy', entropies)


######################################################################################

# Plot of precision and recall with entropies

print('Comienza ploteo-----')
#precision = np.load('precision.npy')
#recall = np.load('recall.npy')
#entropies = np.load('entropies.npy')

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
