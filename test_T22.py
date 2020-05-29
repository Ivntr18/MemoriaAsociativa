import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from matplotlib import cm
import matplotlib as mpl
from associative import AssociativeMemory, AssociativeMemoryError

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
                plt.savefig('T22Audio/grid/amsgrid_'+str(j)+'.png', dpi=500)
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

##Get data
tX=np.load('T22Audio/feat_X.npy')
tY=np.load('T22Audio/feat_Y.npy')
Zeros=tX==0
tX=tX-tX.min()
tX[Zeros]=0
dicY={'a':0,'b':1,'d':2,'e':3,'f':4,'g':5,'i':6,'k':7,'l':8,'m':9,'n':10,'n~':11,'o':12,'p':13,'r':14,'r(':15,'s':16,'t':17,'tS':18,'u':19,'x':20,'Z':21}
tY=[dicY[key] for key in tY]
tY=np.array(tY)
tSize=len(tX)

permutation = np.random.permutation(tSize)
tX=tX[permutation]
tY=tY[permutation]


for ite in range(10):

    test_X=tX[int(ite/10*tSize):int((ite+1)/10*tSize)]
    train_X=np.concatenate((tX[int(0*tSize):int(ite/10*tSize)],tX[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)
    teY=tY[int(ite/10*tSize):int((ite+1)/10*tSize)]
    trY=np.concatenate((tY[int(0*tSize):int(ite/10*tSize)],tY[int((ite+1)/10*tSize):int(1*tSize)]), axis=0)
    # The ranges of all the memories that will be trained
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
    # the domain size. The size of the output layer of the network
    domain = 784
    # the number of categories
    catN=22
    # Maximum value of the features in the train set
    #max_val = train_X.max()

    sel=1
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

    plt.savefig('T22Audio/graph_T22_simple_{0}_{1}.png'.format(sel,ite), dpi=500)
    print('Iteration {0} complete'.format(ite))

    #Do it only once
    break
    #Uncomment the following line for plot at runtime
    #plt.show()
    
# Plot the final graph
'''
average_precision=np.array(average_precision)
main_average_precision=[]

average_recall=np.array(average_recall)
main_average_recall=[]

average_entropy=np.array(average_entropy)
main_average_entropy=[]



for i in range(10):
    main_average_precision.append( average_precision[:,i].mean() )
    main_average_recall.append( average_recall[:,i].mean() )
    main_average_entropy.append( average_entropy[:,i].mean() )
    
print('main avg precision: ',main_average_precision)
print('main avg recall: ',main_average_recall)
print('main avg entropy: ',main_average_entropy)

np.savetxt('main_average_precision--{0}.csv'.format(sel), main_average_precision, delimiter=',')
np.savetxt('main_average_recall--{0}.csv'.format(sel), main_average_recall, delimiter=',')
np.savetxt('main_average_entropy--{0}.csv'.format(sel), main_average_entropy, delimiter=',')

cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])
Z = [[0,0],[0,0]]
step = 0.1
levels = np.arange(0.0, 90 + step, step)
CS3 = plt.contourf(Z, levels, cmap=cmap)

plt.clf()

plt.plot(np.arange(0, 100, 10), main_average_precision, 'r-o', label='Precision')
plt.plot(np.arange(0, 100, 10), main_average_recall, 'b-s', label='Recall')
plt.xlim(-0.1, 91)
plt.ylim(0, 102)
plt.xticks(np.arange(0, 100, 10), sizes)

plt.xlabel('Range Quantization Levels')
plt.ylabel('Percentage [%]')
plt.legend(loc=4)
plt.grid(True)

entropy_labels = [str(e) for e in np.around(main_average_entropy, decimals=1)]

cbar = plt.colorbar(CS3, orientation='horizontal')
cbar.set_ticks(np.arange(0, 100, 10))
cbar.ax.set_xticklabels(entropy_labels)
cbar.set_label('Entropy')

plt.savefig('T22Audio/graph_T22_final_{0}.png'.format(sel), dpi=500)
print("Complete Test")
'''
