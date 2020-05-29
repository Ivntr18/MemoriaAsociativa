import numpy as np
import cupy as cp
from joblib import Parallel, delayed
from associative_gpu import AssociativeMemory, AssociativeMemoryError
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import pickle

average_entropy = []
average_precision = []
average_recall = []

def get_ams_results1(i, s, domain,catN, train_X, test_X, trY, teY, dMatrix,teP):
        table = np.zeros((catN, 5), dtype=np.float64)
        entropy = np.zeros((catN, ), dtype=np.float64)
        ams = dict.fromkeys(range(catN))
        #print(str(ams))
        for j in ams:
            # Create the memories with domain 's'
            ams[j] = AssociativeMemory(domain, s)
        # Round the values
        mean=train_X.mean(axis=0)
        std=train_X.std(axis=0)
        std[std==0]=1
        #train_X=train_X-mean
        #test_X=test_X-mean
        #train_X=train_X/std
        #test_X=test_X/std
        
        
        #valMax=mean/std+3
        valMax=train_X.max(axis=0)
        valMax=2
        #if np.isclose(valMax,0):
        #    valMax=1
        #valMax[np.isclose(valMax,0)]=1
        train_X_around = np.round(train_X * (s - 1) / valMax).astype(np.int32)
        test_X_around = np.round(test_X * (s - 1) / valMax).astype(np.int32)
        train_X_around[train_X_around>(s-1)]=(s-1)
        test_X_around[test_X_around>(s-1)]=(s-1)
        #train_X_around[train_X_around<-(s-1)]=-(s-1)
        #test_X_around[test_X_around<-(s-1)]=-(s-1)
        # Abstraction
        for x, y in zip(train_X_around, trY):
            ams[y].abstract(x, input_range=s)
        print("Abs Done --",s)
        for it in ams:
            with open("Reconocedor/models/AM_"+str(s)+"_"+str(it)+".bin", "bw") as archivo:
                pickle.dump( ams[it], archivo)
        
        # Calculate entropies
        for j in ams:
            #print(j)
            entropy[j] = ams[j].entropy
            #for see grids
            if i==3:
                aspecto=350/s
                plt.imshow( np.flip(cp.asnumpy(ams[j].grid),0) , aspect=aspecto ,cmap='GnBu', origin='lower',alpha=1)
                plt.savefig('T22Audio/amsgrid_'+str(j)+'.png', dpi=500)
        # Reduction
        for x, y, z in zip(test_X_around, teY, teP):
            table[y, 0] += 1
            maxP=0
            sel=-1
            for k in ams:
                if ams[k].reduce(x, input_range=s):
                    if maxP < dMatrix[k,z]:
                        maxP=dMatrix[k,z]
                        sel=k
            if sel==y:
                #TP
                table[y, 1] += 1
                for j in range(22):
                    if j!=y:
                        #TN
                        table[j, 3] += 1
            else:
                if sel!=-1:
                    #FP
                    table[sel,2] += 1
                #FN
                table[y, 4] += 1
                
                for j in range(22):
                    if j!=sel and j!=y:
                        #TN
                        table[j, 3] += 1
            
            
                    
                    
        print("Reduc Done --",s)
        return (i, table, entropy)


ite=0
##Get data
test_X=np.load('T22Audio/test_features_X.npy')
train_X=np.load('T22Audio/train_features_X.npy')
teY=np.load('T22Audio/test_features_Y.npy')
trY=np.load('T22Audio/train_features_Y.npy')

##Only positive
test_X+=1
train_X+=1

dMatrix=np.load("T22Audio/matDis.npy")
test_prev=np.load('T22Audio/test_prev.npy')

#Compute prob P(St)
Pst=dMatrix.sum(axis=1)/dMatrix.sum()
##Compute prob P(St|St-1)
dMatrix=dMatrix/dMatrix.sum(axis=0)

#P(St|St-1)*(1/(10*P(St)))
dMatrix=dMatrix.T*Pst
dMatrix=dMatrix.T

#test_X=np.load('test_features_l4.npy')
#train_X=np.load('train_features_l4.npy')
#teY=np.load('test_features_l4Y.npy')
#trY=np.load('train_features_l4Y.npy')
#print(teY)

#np.cuda.Device(0)
#np.cuda.Stream.null.synchronize()


    # The ranges of all the memories that will be trained
sizes = (16,32,64,128,256,512,1024)
#sizes = (4,128)
    # the domain size. The size of the output layer of the network
domain = 600
    # the number of categories
catN=22
    # Maximum value of the features in the train set
    #max_val = train_X.max()

    # Train the different co-domain memories
    
tables = np.zeros((len(sizes), catN, 5), dtype=np.float64)
entropies = np.zeros((len(sizes), int(catN)), dtype=np.float64)


print('Train the different co-domain memories -----',ite)
#for i, s in enumerate(sizes):
    #list_tables_entropies=get_ams_results1(i, s, domain, train_X, test_X, trY, teY)
list_tables_entropies = Parallel(n_jobs=8, verbose=50)(
    delayed(get_ams_results1)(i, s, domain,catN, train_X, test_X, trY, teY,dMatrix,test_prev) for i, s in enumerate(sizes))

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
    Nnan=np.isnan(prec_aux).sum()
    prec_aux[np.isnan(prec_aux)]=0
    recall_aux = tables[i, :, 1] / (tables[i, :, 1] + tables[i, :, 4])
    precision[i, 0:catN, 0] = prec_aux[:]
    #precision[i, catN, 0] = prec_aux.mean()
    precision[i, catN, 0] = prec_aux.sum()/(catN-Nnan)
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


plt.plot(np.arange(0, 100, 100/len(sizes)), average_precision[ite], 'r-o', label='Precision')
plt.plot(np.arange(0, 100, 100/len(sizes)), average_recall[ite], 'b-s', label='Recall')
plt.xlim(-0.1, 91)
plt.ylim(0, 102)
plt.xticks(np.arange(0, 100, 100/len(sizes)), sizes)

plt.xlabel('Range Quantization Levels')
plt.ylabel('Percentage [%]')
plt.legend(loc=4)
plt.grid(True)

entropy_labels = [str(e) for e in np.around(average_entropy[ite], decimals=1)]

cbar = plt.colorbar(CS3, orientation='horizontal')
cbar.set_ticks(np.arange(0, 100, 100/len(sizes)))
cbar.ax.set_xticklabels(entropy_labels)
cbar.set_label('Entropy')

plt.savefig('T22Audio/graph_T22_{0}.png'.format(ite), dpi=500)
print('Iteration {0} complete'.format(ite))
