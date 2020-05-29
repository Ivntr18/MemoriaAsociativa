import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap 
from joblib import Parallel, delayed
from mnist import load_mnist
from associative import AssociativeMemory, AssociativeMemoryError
import sys

mnist_path = './mnist'

def get_ams_results1(s, domain, train_X, test_X, trY, teY):
        table = np.zeros((10, 5), dtype=np.float64)
        entropy = np.zeros((10, ), dtype=np.float64)
        ams = dict.fromkeys(range(10))
        #print(str(ams))
        for j in ams:
            # Create the memories with domain 's'
            ams[j] = AssociativeMemory(domain, s)
        # Round the values
        if train_X.max()>test_X.max():
            valMax=train_X.max()
        else:
            valMax=test_X.max()
        train_X_around = np.round(train_X * (s - 1) / valMax).astype(np.int16)
        test_X_around = np.round(test_X * (s - 1) / valMax).astype(np.int16)
        # Abstraction
        print('start abstraction')
        for x, y in zip(train_X_around, trY):
            ams[y].abstract(x, input_range=s)
        print('end abstraction')

        return (ams)

    
def get_ams_results2( s, domain, train_X, test_X, trY, teY):
        table = np.zeros((10, 5), dtype=np.float64)
        entropy = np.zeros((5, ), dtype=np.float64)
        ams = dict.fromkeys(range(5))
        #print(str(ams))
        for j in ams:
            # Create the memories with domain 's'
            ams[j] = AssociativeMemory(domain, s)
        # Round the values
        if train_X.max()>test_X.max():
            valMax=train_X.max()
        else:
            valMax=test_X.max()
        train_X_around = np.round(train_X * (s - 1) / valMax).astype(np.int16)
        test_X_around = np.round(test_X * (s - 1) / valMax).astype(np.int16)
        # Abstraction
        print('start abstraction')
        for x, y in zip(train_X_around, trY):
            yy=y%5
            ams[yy].abstract(x, input_range=s)
        print('end abstraction')
        
        return (ams)

def main(size, num):
    ite=0
    train_X = np.load('train_features_l4.npy')
    test_X = np.load('test_features_l4.npy')
    trX, teX, trY, teY = load_mnist(mnist_path, onehot=False)
    tX=np.concatenate((trX, teX), axis=0)
    tY=np.concatenate((trY, teY), axis=0)
    tSize=len(tX)
    teX=tX[int(ite/10*tSize):int((ite+1)/10*tSize)]
    trX=np.concatenate((tX[int(0*tSize):int(ite/10*tSize)],tX[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)
    teY=tY[int(ite/10*tSize):int((ite+1)/10*tSize)]
    trY=np.concatenate((tY[int(0*tSize):int(ite/10*tSize)],tY[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)
    # The ranges of all the memories that will be trained
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    #sizes = (1, 2, 4, 8, 16)
    # the domain size. The size of the output layer of the network
    domain = 625
    # Maximum value of the features in the train set
    max_val = train_X.max()
    sel=1


    # Train the different co-domain memories

    tables = np.zeros((len(sizes), 10, 5), dtype=np.float64)
    entropies = np.zeros((len(sizes), int(10/sel)), dtype=np.float64)
    memories_set=np.zeros(len(sizes))

    
    print('Train the different co-domain memories -- NinM: ',sel,' -----',ite)
    if sel == 1:
        memories=get_ams_results1 ( size, domain, train_X, test_X, trY, teY)
    elif sel == 2:
        memories=get_ams_results2 ( size, domain, train_X, test_X, trY, teY)
    else:
        print('Error sel')
        
    print(memories[num].grid)
    plt.figure()
    plt.imshow(memories[num].grid,aspect=10,cmap='tab20')
    plt.savefig('memory.png'.format(sel), dpi=500)



if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(int(sys.argv[1]),int(sys.argv[2]))
    else:
        print ("Uso: python3 graph_memories.py size num")
