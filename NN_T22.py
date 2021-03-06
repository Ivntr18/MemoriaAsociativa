import theano
import numpy as np
import matplotlib.pyplot as plt
from theano import tensor as T
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl
from convnet import init_weights, RMSprop, convnet_model
from associative import AssociativeMemory, AssociativeMemoryError
from sklearn.preprocessing import OneHotEncoder

average_entropy = []
average_precision = []
average_recall = []

def get_ams_results1(i, s, domain, train_X, test_X, trY, teY):
        catN=22
        table = np.zeros((catN, 5), dtype=np.float64)
        entropy = np.zeros((catN, ), dtype=np.float64)
        ams = dict.fromkeys(range(catN))
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
        for x, y in zip(train_X_around, trY):
            ams[y].abstract(x, input_range=s)
        # Calculate entropies
        for j in ams:
            #print(j)
            entropy[j] = ams[j].entropy
            #for see grids
            if i==3:
                aspecto=550/8
                plt.imshow( np.flip(ams[j].grid,0) , aspect=aspecto ,cmap='GnBu', origin='lower',alpha=1)
                plt.savefig('T22Audio/amsgrid_'+str(j)+'.png', dpi=500)
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

tX=np.load('T22Audio/feat_X.npy')
tY=np.load('T22Audio/feat_Y.npy')
Zeros=tX==0
offset=tX.min()
tX=tX-tX.min()
#tX[Zeros]=0
dicY={'a':0,'b':1,'d':2,'e':3,'f':4,'g':5,'i':6,'k':7,'l':8,'m':9,'n':10,'n~':11,'o':12,'p':13,'r':14,'r(':15,'s':16,'t':17,'tS':18,'u':19,'x':20,'Z':21}
tY=[dicY[key] for key in tY]
tY_encOFF=np.array(tY)
enc = OneHotEncoder()
enc.fit(tY_encOFF.reshape(-1,1))
tY=enc.transform(tY_encOFF.reshape(-1,1)).toarray()
tSize=len(tX)

permutation = np.random.permutation(tSize)
tX=tX[permutation]
tY=tY[permutation]
tY_encOFF=tY_encOFF[permutation]

print(tY.shape)

X=T.ftensor4()
Y=T.fmatrix()

##Selector of experiment for now only 1
sel=1

##for ite in range(10):
ite=0
    
teX=tX[int(ite/10*tSize):int((ite+1)/10*tSize)]
trX=np.concatenate((tX[int(0*tSize):int(ite/10*tSize)],tX[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)

teY=tY[int(ite/10*tSize):int((ite+1)/10*tSize)]
trY=np.concatenate((tY[int(0*tSize):int(ite/10*tSize)],tY[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)

print("Length Train {0}".format(len(trX)))
print("Length Test {0}".format(len(teX)))


#trX = trX.reshape(-1, 1, 28, 28)
#teX = teX.reshape(-1, 1, 28, 28)
trX = trX.reshape(-1, 1, 13, 13)
teX = teX.reshape(-1, 1, 13, 13)

auxtrX=np.zeros((trX.shape[0],1,28,28))
auxteX=np.zeros((teX.shape[0],1,28,28))

auxtrX=auxtrX-offset
auxteX=auxteX-offset


auxtrX[:,:,:13,:13]=trX
auxteX[:,:,:13,:13]=teX

trX=auxtrX
teX=auxteX




    
    
################################################################

# Setup of the training and testing.
# Do not run this cell if you already have the weights of the network. 

#w1 = init_weights((32, 1, 3, 3))
w1 = init_weights((32, 1, 3, 3))
#w2 = init_weights((64, 32, 3, 3))
w2 = init_weights((64, 32, 3, 3))
#w3 = init_weights((128, 64, 3, 3))
w3 = init_weights((128, 64, 3, 3))
#w4 = init_weights((128 * 3 * 3, 625))
w4=init_weights((128 * 3 * 3, 625))
w5 = init_weights((625, 22))

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
    #for start, end in zip(range(0, len(trX), 64), range(128, len(trX), 64)):
        cost = train(trX[start:end], trY[start:end])
    print('Testing epoch number: ',i,' -----',ite)
    #print(np.mean(np.argmax(teY, axis=1) == predict(teX)))

# Save the weights of the network


np.save('T22Audio/w1_t22.npy', w1.get_value())
np.save('T22Audio/w2_t22.npy', w2.get_value())
np.save('T22Audio/w3_t22.npy', w3.get_value())
np.save('T22Audio/w4_t22.npy', w4.get_value())
np.save('T22Audio/w5_t22.npy', w5.get_value())
    
    #del train
    #del cost


    ##################################################################################

    # Load the network's parameters to generate the features
    # Do not run this cell if you already generated the features. Skip to the next cell

# Shared variables
w1 = theano.shared(np.load('T22Audio/w1_t22.npy'), name='w1')
w2 = theano.shared(np.load('T22Audio/w2_t22.npy'), name='w2')
w3 = theano.shared(np.load('T22Audio/w3_t22.npy'), name='w3')
w4 = theano.shared(np.load('T22Audio/w4_t22.npy'), name='w4')
w5 = theano.shared(np.load('T22Audio/w5_t22.npy'), name='w5')

    # model
l1, l2, l3, l4, py_x = convnet_model(X, w1, w2, w3, w4, w5, 0., 0.)

generate = theano.function(inputs=[X], outputs=l4, allow_input_downcast=True)

    # Generate features from the network 128*3*3->625

train_features = np.zeros((len(trX), (625)), theano.config.floatX)

for start, end in zip(range(0, len(trX), 200), range(200, (len(trX) + 1), 200)):
    batch = generate(trX[start:end])
    train_features[start:end] = batch
    
np.save('T22Audio/train_features_X.npy', train_features)


test_features = np.zeros((len(teX), (625)), theano.config.floatX)

for start, end in zip(range(0, len(teX), 200), range(200, (len(teX) + 1), 200)):
    batch = generate(teX[start:end])
    test_features[start:end] = batch
    
np.save('T22Audio/test_features_X.npy', test_features)

tY=tY_encOFF
teY=tY[int(ite/10*tSize):int((ite+1)/10*tSize)]
trY=np.concatenate((tY[int(0*tSize):int(ite/10*tSize)],tY[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)

np.save('T22Audio/test_features_Y.npy', teY)
np.save('T22Audio/train_features_Y.npy', trY)

##Part II



##Get data
test_X=np.load('T22Audio/test_features_X.npy')
train_X=np.load('T22Audio/train_features_X.npy')
teY=np.load('T22Audio/test_features_Y.npy')
trY=np.load('T22Audio/train_features_Y.npy')



    # The ranges of all the memories that will be trained
sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    # the domain size. The size of the output layer of the network
domain = 625
    # the number of categories
catN=22
    # Maximum value of the features in the train set
    #max_val = train_X.max()

    # Train the different co-domain memories
    
tables = np.zeros((len(sizes), catN, 5), dtype=np.float64)
entropies = np.zeros((len(sizes), int(catN/sel)), dtype=np.float64)


print('Train the different co-domain memories -- NinM: ',sel,' -----',ite)
if sel == 1:
        #for i, s in enumerate(sizes):
            #list_tables_entropies=get_ams_results1(i, s, domain, train_X, test_X, trY, teY)
    list_tables_entropies = Parallel(n_jobs=8, verbose=50)(
        delayed(get_ams_results1)(i, s, domain, train_X, test_X, trY, teY) for i, s in enumerate(sizes))
elif sel == 2:
    list_tables_entropies = Parallel(n_jobs=8, verbose=50)(
        delayed(get_ams_results2)(i, s, domain, train_X, test_X, trY, teY) for i, s in enumerate(sizes))

for i, table, entropy in list_tables_entropies:
    tables[i, :, :] = table
    entropies[i, :] = entropy
    
np.save('T22Audio/tables.npy', tables)
np.save('T22Audio/entropies.npy', entropies)

    # Table columns
    # 0.- Total count
    # 1.- Able to reduce and it is the same number
    # 2.- Able to reduce and it is not the same number
    # 3.- Not able to reduce and it is not the same number
    # 4.- Not able to reduce and it is the same number

    ##########################################################################################

    # Calculate the precision and recall

print('Calculate the precision and recall')
precision = np.zeros((len(sizes), catN+1, 1), dtype=np.float64)
recall = np.zeros((len(sizes), catN+1, 1), dtype=np.float64)

for i, s in enumerate(sizes):
    prec_aux = tables[i, :, 1] / (tables[i, :, 1] + tables[i, :, 2])
    recall_aux = tables[i, :, 1] / tables[i, :, 0]
    precision[i, 0:catN, 0] = prec_aux[:]
    precision[i, catN, 0] = prec_aux.mean()
    recall[i, 0:catN, 0] = recall_aux[:]
    recall[i, catN, 0] = recall_aux.mean()
    

    ######################################################################################

    # Plot of precision and recall with entropies

print('Plot of precision and recall with entropies-----{0}'.format(ite))
average_entropy.append( entropies.mean(axis=1) )
    # Percentage
average_precision.append( precision[:, catN, :] * 100 )
average_recall.append( recall[:, catN, :] * 100 )
    
np.save('average_precision.npy', average_precision)
np.save('average_recall.npy', average_recall)
np.save('average_entropy.npy', average_entropy)
    
print('avg precision: ',average_precision[ite])
print('avg recall: ',average_recall[ite])
print('avg entropy: ',average_entropy[ite])

    # Setting up a colormap that's a simple transtion
cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])

    # Using contourf to provide my colorbar info, then clearing the figure
Z = [[0,0],[0,0]]
step = 0.1
levels = np.arange(0.0, 90 + step, step)
CS3 = plt.contourf(Z, levels, cmap=cmap)

plt.clf()


plt.plot(np.arange(0, 100, 10), average_precision[ite], 'r-o', label='Precision')
plt.plot(np.arange(0, 100, 10), average_recall[ite], 'b-s', label='Recall')
plt.xlim(-0.1, 91)
plt.ylim(0, 102)
plt.xticks(np.arange(0, 100, 10), sizes)

plt.xlabel('Range Quantization Levels')
plt.ylabel('Percentage [%]')
plt.legend(loc=4)
plt.grid(True)

entropy_labels = [str(e) for e in np.around(average_entropy[ite], decimals=1)]

cbar = plt.colorbar(CS3, orientation='horizontal')
cbar.set_ticks(np.arange(0, 100, 10))
cbar.ax.set_xticklabels(entropy_labels)
cbar.set_label('Entropy')

plt.savefig('T22Audio/graph_T22_{0}_{1}.png'.format(sel,ite), dpi=500)
print('Iteration {0} complete'.format(ite))


