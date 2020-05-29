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
    
_, _, trY, teY = load_mnist(mnist_path, onehot=False)
np.save('train_features_l4.npy', train_features)
np.save('train_features_l4Y.npy', trY)

test_features = np.zeros((10000, (625)), theano.config.floatX)

for start, end in zip(range(0, len(teX), 200), range(200, (len(teX) + 1), 200)):
    print(start, end)
    batch = generate(teX[start:end])
    test_features[start:end] = batch
    
np.save('test_features_l4.npy', test_features)
np.save('test_features_l4Y.npy', teY)


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
    
np.save('precision.npy', precision)
np.save('recall.npy', recall)
np.save('entropies.npy', entropies)