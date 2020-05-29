
# coding: utf-8

# In[1]:


import glob
import subprocess
import scipy.io.wavfile as wav
import scipy.signal
import numpy as np
import csv
from python_speech_features import mfcc
import pickle
import keras
#from tensorflow.keras import backend as K
from keras.layers import *
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.preprocessing import sequence
import difflib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt

sample_rate, signal = wav.read("s01002.wav")

dicY=['a','b','d','e','f','g','i','k','l','m','n','n~','o','p','r','r(','s','t','tS','u','x','Z']
dicYFull={'a':0,'b':1,'d':2,'e':3,'f':4,'g':5,'i':6,'k':7,'l':8,'m':9,'n':10,'n~':11,'o':12,'p':13,'r':14,'r(':15,'s':16,'t':17,'tS':18,'u':19,'x':20,'Z':21}


# In[2]:


models=[]
s=512
for i in range(22):
    with open("models/AM_"+str(s)+"_"+str(i)+".bin", "rb") as f:
        dump = pickle.load(f)
        models.append(dump)


# In[3]:


# load YAML and create model
yaml_file = open('models/modelConv.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
modelConv = model_from_yaml(loaded_model_yaml)
# load weights into new model
modelConv.load_weights("models/modelConv.h5")
print("Loaded model from disk")


# In[4]:


def predict(ns,s):
    feat=mfcc(ns,16000,numcep=26)
    vector=feat.flatten()
    vector-=-86
    vectorPad=sequence.pad_sequences([vector],338)
    lstmVect=modelConv.predict(vectorPad)
    #print(vectorPad)
    lstmVect+=1
    valMax=2
    lstmVect_around = np.round(lstmVect * (s - 1) / valMax).astype(np.int32)
    #print(lstmVect_around)
    res=[]
    for i,k in enumerate(models):
        #print("R")
        if k.reduce(lstmVect_around, input_range=s):
            res.append(i)
            #print("REDUCE")
    return res


# In[5]:


def reduce_signal(signal,sample_rate):
    listOfSol=[]
    sol=[]
    inicio=0
    step=1 #0.5
    
    while True:
        inicio+=step
        dur=63.19+(np.random.rand()-0.5)*2*26
        final=inicio+dur
        ns=signal[int(float(inicio)/1000 * sample_rate):int(float(final)/1000 * sample_rate)]
        #print(ns.shape,sample_rate)
        if sample_rate!=16000 and len(ns)!=0:
            sampls=int(dur/1000*16000)
            ns=scipy.signal.resample(ns,sampls)
            #print(ns.shape, dur)
        else:
            print("ELSE---")
            break
        sol=predict(ns,s)
        for i,it in enumerate(sol):
            sol[i]=dicY[it]
        if sol:
            media=(inicio+final)/2
            listOfSol.append((sol,media))
            #print(sol, media)
    return listOfSol
    


# In[6]:


X=np.array([[1,2,3,4,5,6],[7,8,9,10,11,12],[13,14,15,16,17,18],[19,20,21,22,23,24]])
Y=np.array([[25,26,27,28,29,30],[31,32,33,34,35,36],[37,38,39,40,41,42]])
Z=np.array([100,105,1561,516,5615,-5616,5165,15])

L=[]
X=X.flatten()
Y=Y.flatten()
L.append(X)
L.append(Y)
L.append(Z)
L=np.array(L)


# In[7]:


def mostFrecuency(buffer,min_elements):
    L=[]
    for i in buffer:
        for j in i:
            L.append(j)
    temp=L.copy()
    for i in range(len(L)):
        H=L.pop()
        if L.count(H)==0 and temp.count(H)>=min_elements:
            L.insert(0,H)
    L.sort(key=temp.count, reverse=True)
    return L[:3]


# In[8]:


def frecuency_filter(listOfSol):
    buf=[]
    output=[]
    buf_size=10 #20
    min_elements=7 #15
    for elemOfSol in listOfSol:
        it=elemOfSol[0]
        enum=elemOfSol[1]
        buf.insert(0,it)
        if len(buf)>buf_size:
            buf.pop()
        if len(buf)>=buf_size:
            #print(val)
            if not(output):
                frec=mostFrecuency(buf,min_elements)
                buf.clear()
                if frec:
                    output.append((frec,enum))
                    print("not output")
                    #buf.clear()
            #elif val>0.99 and output[len(output)-1]!=mostFrecuency(buf):
            else:
                frecList=mostFrecuency(buf,min_elements)
                pre_out=[]
                if frecList:
                    for i,frecEl in enumerate(frecList):
                        if output[len(output)-1][0].count(frecEl)==0:
                            #print("IFFF",frecEl,i)
                            if i<3:
                                #print("***1",enum)
                                pre_out.append(frecEl)
                            elif pre_out:
                                #print("***2")
                                pre_out.append(frecEl)
                            else:
                                #print("***3",enum)
                                output[len(output)-1][0].append(frecEl)
                if pre_out:
                    output.append((pre_out,enum))
                    buf.clear()

    return output
            
            
        
    
        
    
    


# In[9]:


def get_regular_exp(wordFile,phonFile,red_signal):
    regExpList=[]
    init=0
    filt_signal=frecuency_filter(red_signal)
    #print(filt_signal)
    with open(wordFile) as csvfile,open(phonFile) as csvfile2 :
        readerWord=csv.reader(csvfile, delimiter=' ')
        readerPhon=csv.reader(csvfile2, delimiter=' ')
        #pointW=2
        #pointP=2
        phonL=[]
        wordL=[]
        outputL=[]
        
        nextP=next(readerPhon)

        nextW=next(readerWord)
        init=0
        final=0

        for it in filt_signal:
            while True:
                if nextP[0]=='MillisecondsPerFrame:' or nextP[0]=='END':
                    nextP=next(readerPhon)
                    continue
                final=float(nextP[1])
                #print(nextW)
                while True:
                    try:
                        if nextW[2]!='.bn' and nextW[2]!='.sil' and nextW[2]!='HEADER':
                            break
                    except:
                        pass
                    try:
                        nextW=next(readerWord)
                    except:
                        print("BREAK 1")
                        break
                #print(str(final),nextW[1])
                if final>float(nextW[1]):
                    #print("IFF",nextW[2],wordL)
                    outputL.append((nextW[2],wordL))
                    wordL=[]
                    phonL=[]
                    nextW=next(readerWord)


                
                if nextP[2]!='.bn' and nextP[2]!='.sil':
                    if float(it[1])>init and float(it[1])<final:
                        #print("append",phonL,it[0],it[1])
                        for el in it[0]:
                            phonL.append(el)
                        break
                    elif phonL:
                        wordL.append(phonL)
                        #print("clear",it[0],it[1])
                        phonL=[]
                    else:
                        wordL.append(['-'])

                init=final
                #wordL.append(phonL)
                try:
                    nextP=next(readerPhon)
                except:
                    print("BREAK 2")
                    break
        wordL.append(phonL)
        for i in range(len(wordL),len(nextW[2].replace("_7",""))):
            wordL.append(['-'])
        outputL.append((nextW[2],wordL))
         
     
    
    return outputL


# In[10]:



def get_regular_exp_old(wordFile,red_signal):
    regExpList=[]
    init=0
    filt_signal=frecuency_filter(red_signal)
    #print(filt_signal)
    with open(wordFile) as csvfile:
        reader=csv.reader(csvfile, delimiter=' ')
        for i,row in enumerate(reader):
            if row and row[0]!='MillisecondsPerFrame:' and row[0]!='END':
                if (row[2] == '.sil') or (row[2] == '.bn'):
                    init=float(row[1])
                else:
                    word=row[2]
                    final=float(row[1])
                    regExp=''
                    
                    for it in filt_signal:
                        if float(it[1])>init and float(it[1])<final:
                            if len(it[0])>1:
                                regExp+='['
                            for count,elem in enumerate(it[0]):
                                #if count >0:
                                #    regExp+='+'
                                regExp+=elem
                            if len(it[0])>1:
                                regExp+=']'
                            #regExp+='*'
                    init=float(row[1])
                    buf=''
                    bufWR=''
                    tempRegExp=regExp
                    for letter in tempRegExp:
                        if (letter=='a' or letter=='e' or letter=='i' or letter=='o' or letter=='u'):
                            if letter in buf:
                                print("111")
                                buf+=letter
                            else:
                                buf+=letter
                                bufWR+=letter
                        else:
                            if len(buf)>1:
                                bufString='['
                                for let in bufWR:
                                    bufString+=let
                                bufString+=']'
                                #print("REPLACE",buf,bufString)
                                regExp=regExp.replace(buf,bufString)
                                buf=''
                                bufWR=''
                            else:
                                buf=''
                                bufWR=''
                    if len(buf)>1:
                        bufString='['
                        for let in bufWR:
                            bufString+=let
                        bufString+=']'
                        #print("REPLACE",buf,bufString)
                        regExp=regExp.replace(buf,bufString)
                                
                        
                    regExpList.append((regExp,word))
     
    
    return regExpList
                


# In[11]:


def get_graphs(phonFile,red_signal):
    #regExpList=[]
    #init=0
    #filt_signal=frecuency_filter(red_signal)
    #print(filt_signal)
    #0 TP
    #1 FP
    #2 TN
    #3 FN
    confMat=np.zeros((22,4))
    with open(phonFile) as csvfile:
        #readerWord=csv.reader(csvfile, delimiter=' ')
        readerPhon=csv.reader(csvfile, delimiter=' ')
        #pointW=2
        #pointP=2
        phonL=[]
        #wordL=[]
        outputL=[]
        
        nextP_=next(readerPhon)
        #print(nextP)
        

        #nextW=next(readerWord)
        init=0
        final=0
        listP=[]
        while True:
            if nextP_[0]=='MillisecondsPerFrame:' or nextP_[0]=='END':
                nextP_=next(readerPhon)
                #print(nextP)
                continue
            listP.append(nextP_)
            try:
                nextP_=next(readerPhon)
            except:
                break
                

        for it in red_signal:
            init=0
            final=0
            for nextP in listP:
                final=float(nextP[1])
                phon=nextP[2]
                if phon!='.bn' and phon!='.sil':
                    if float(it[1])>init and float(it[1])<final:
                        for el in it[0]:
                            if el==phon:
                                confMat[dicYFull[phon],0]+=1
                                for i in range(22):
                                    if i!=dicYFull[phon]:
                                        confMat[i,2]+=1
                            else:
                                confMat[dicYFull[phon],3]+=1
                                confMat[dicYFull[el],1]+=1
                                for i in range(22):
                                    if i!=dicYFull[phon] and i!=dicYFull[el]:
                                        confMat[i,2]+=1
                                
                        print("TRueBReak", it)  
                    else:
                        pass
                init=final
                
                
            
                    
            
                #wordL.append(phonL)
    return confMat

def pltgraph(confMat):
                    
    prec=confMat[:,0]/(confMat[:,0]+confMat[:,1])*100
    recall=confMat[:,0]/(confMat[:,0]+confMat[:,3])*100
    print("PREC Y RECALL")
    
    cmap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['cyan','purple'])

    # Using contourf to provide my colorbar info, then clearing the figure
    Z = [[0,0],[0,0]]
    step = 0.1
    levels = np.arange(0.0, 90 + step, step)
    CS3 = plt.contourf(Z, levels, cmap=cmap)

    plt.clf()


    plt.plot(np.arange(0, 22, 1), prec, 'r-o', label='Precision')
    plt.plot(np.arange(0, 22, 1), recall, 'b-s', label='Recall')
    plt.xlim(-0.1, 22)
    plt.ylim(0, 100)
    #plt.xticks(np.arange(0, 22, 1), 22)

    plt.xlabel('Range Quantization Levels')
    plt.ylabel('Percentage [%]')
    plt.legend(loc=4)
    plt.grid(True)


    #cbar = plt.colorbar(CS3, orientation='horizontal')
    #cbar.set_ticks(np.arange(0, 22, 1))
    

    plt.savefig('graph.png', dpi=500)     
     
    
    


# redS=[]
# t22file=''
# phonFile=''
# outputAll=[]
# count=0
# for t22file in glob.iglob('../DIMEX100/CorpusDimex100/**/tp/comunes/*.phn',recursive=False):
#     #Se obtiene el prefijo del archivo
#     split=t22file.split("/")
#     folder=split[3]
#     #print(split)
#     subfolder=split[5]
#     name=split[6]
#     name_split=name.split(".")
#     #name_split=t22file.split(".")
#     name=name_split[0]
#     #Se busca por su correspondiente archivo de audio y marcas fonéticas
#     phonFile='../DIMEX100/CorpusDimex100/'+folder+'/T22/'+subfolder+'/'+name+'.phn'
#     ls=glob.glob('../DIMEX100/CorpusDimex100/'+folder+'/audio_editado/'+subfolder+'/'+name+'.wav',recursive=False)
#     print(t22file,ls)
#     try:
#         audiofile=ls[0]
#     except:
#         print("ERROR: "+folder+name)
#         continue
#     #se lee el archivo de audio .wav
#     try:
#         sample_rate, signal = wav.read(audiofile)
#     except:
#         print("AUDIOFILE ERROR")
#         continue
#     redS=reduce_signal(signal,sample_rate)
#     print(redS)
#     continue
#     regList=get_regular_exp(t22file,phonFile,redS)
#     outputAll.append(regList)
#     #print(redS)
#     print(get_regular_exp_old(t22file,redS))
#     print(regList)
#     
#     count+=1
#     if count>600:
#         break
#     #break
#     
#     
# with open("outL.bin", "wb") as fp:   #Pickling   
#     pickle.dump(outputAll, fp)                
#     
#     
#     
#         

# In[ ]:


redS=[]
t22file=''
phonFile=''
outputAll=[]
count=0
cM=np.zeros((22,4))
for t22file in glob.iglob('../DIMEX100/CorpusDimex100/**/tp/comunes/*.phn',recursive=False):
    #Se obtiene el prefijo del archivo
    split=t22file.split("/")
    folder=split[3]
    #print(split)
    subfolder=split[5]
    name=split[6]
    name_split=name.split(".")
    #name_split=t22file.split(".")
    name=name_split[0]
    #Se busca por su correspondiente archivo de audio y marcas fonéticas
    phonFile='../DIMEX100/CorpusDimex100/'+folder+'/T22/'+subfolder+'/'+name+'.phn'
    ls=glob.glob('../DIMEX100/CorpusDimex100/'+folder+'/audio_editado/'+subfolder+'/'+name+'.wav',recursive=False)
    print(t22file,ls)
    try:
        audiofile=ls[0]
    except:
        print("ERROR: "+folder+name)
        continue
    #se lee el archivo de audio .wav
    try:
        sample_rate, signal = wav.read(audiofile)
    except:
        print("AUDIOFILE ERROR")
        continue
    redS=reduce_signal(signal,sample_rate)
    #print(redS)
    
    cM=cM+get_graphs(phonFile,redS)
    pltgraph(cM)
    print("plot cm",cM)
    #outputAll.append(regList)
    #print(redS)
    #print(get_regular_exp_old(t22file,redS))
    #print(regList)
    
    
    count+=1
    print(count)
    if count>600:
        break
    #break
    
     
    


# In[ ]:


with open('outL.bin', 'r') as file:   # 'r' for reading; can be omitted
    mydict = pickle.load(file)         # load file content as mydict
    file.close()                       

print(mydict)


# In[ ]:





