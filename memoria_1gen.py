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









