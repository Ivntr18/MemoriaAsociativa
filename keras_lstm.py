import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
#from tensorflow.keras import backend as K
from keras.layers import *
from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.models import model_from_yaml

batch_size = 128
num_classes = 22
epochs = 8
img_rows, img_cols = 1, 1
np.random.seed(5)


#X=np.load('T22Audio/feat_X.npy')
#Y=np.load('T22Audio/feat_Y.npy')
X=np.load('DIMEX100/Features/feat_X.npy')
Y=np.load('DIMEX100/Features/feat_Y.npy')
#X=np.array(rawX[:,:img_rows * img_cols], dtype='float32')
#Prev=np.load('T22Audio/prevL.npy')
Prev=np.load('DIMEX100/Features/prevL.npy')
#Zeros=X==0
#offset=X.min()
#X=X-X.min()
#X[Zeros]=0
#X=X.reshape(X.shape[0],img_rows, img_cols)
dicY={'a':0,'b':1,'d':2,'e':3,'f':4,'g':5,'i':6,'k':7,'l':8,'m':9,'n':10,'n~':11,'o':12,'p':13,'r':14,'r(':15,'s':16,'t':17,'tS':18,'u':19,'x':20,'Z':21}
dicPrev={'-':-1,'a':0,'b':1,'d':2,'e':3,'f':4,'g':5,'i':6,'k':7,'l':8,'m':9,'n':10,'n~':11,'o':12,'p':13,'r':14,'r(':15,'s':16,'t':17,'tS':18,'u':19,'x':20,'Z':21}
Y=[dicY[key] for key in Y]
Y=np.array(Y, dtype='uint8')
Prev=[dicPrev[key] for key in Prev]
Prev=np.array(Prev, dtype='uint8')
Prev=Prev+1
sizeInp=X.shape[0]
permutation = np.random.permutation(sizeInp)
X=X[permutation]
Y=Y[permutation]
Prev=Prev[permutation]

#####
minX=sequence.pad_sequences(X).min()
X-=minX
print(minX)
X=sequence.pad_sequences(X)
#normalizacion
#mean=X.mean()
#std=X.std()
#X=X-mean
#X=X/std
#X=X/X.max()
img_cols=X.shape[1]

#X=X.reshape(X.shape[0],int(X.shape[1]/26),26)


x_train=X[:int(0.9*sizeInp)]
x_test=X[int(0.9*sizeInp):]
y_train=Y[:int(0.9*sizeInp)]
y_test=Y[int(0.9*sizeInp):]
Prev_te=Prev[int(0.9*sizeInp):]


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#x_train /= X.max()
#x_test /= X.max()

#Save y_train/test before onehot encoding
np.save('T22Audio/train_features_Y.npy',y_train)
np.save('T22Audio/test_features_Y.npy',y_test)
np.save('T22Audio/test_prev.npy',Prev_te)
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

embed_dim = 256
lstm_out = 600
input_shape=(X.shape[1],)

def Conv():
    model_lstm = Sequential()
    model_lstm.add(Embedding(X.max(), embed_dim,input_length = X.shape[1]))
    model_lstm.add(CuDNNLSTM(lstm_out))
    #print(model_lstm.summary())
    return model_lstm


def FCN():
    model_fc = Sequential()
    model_fc.add(Dense(num_classes, activation='softmax'))
    #input_net = Input(shape=(lstm_out,))  # adapt this if using `channels_first` image data format   
    return model_fc

x = Input(shape=(input_shape))
modelFCN=FCN()
modelConv=Conv()
# make the model:
model = Model(x, modelFCN(modelConv(x)))
print(model.summary())


model.compile(keras.optimizers.Adam(),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

np.save('T22Audio/test_features_X.npy',modelConv.predict(x_test))
np.save('T22Audio/train_features_X.npy',modelConv.predict(x_train))

model_yaml = modelConv.to_yaml()
with open("Reconocedor/models/modelConv.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
modelConv.save_weights("Reconocedor/models/modelConv.h5")
print("Saved model to disk")
