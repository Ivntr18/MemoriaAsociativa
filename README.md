# Memoria Asociativa

## Preproceso del corpus

El primer paso es correr el script DIMEX100/gaus_t22.py
este script creara los archivos Features/media.npy Features/std.npy y Features/matDis.npy

Posteriormente se ejecuta el script prepro_corpus-Dimex.py
el cual obtiene los features mfcc de cada muestra y las almacena en Features/feat_X.npy Features/feat_Y.npy y Features/prevL.npy

## Entrenamiento de red neuronal

El script keras_lstm.py se encarga de entrenar una red neuronal recurrente con los features mfcc, su salida es un nuevo featur de 600 elementos el cual se almacena en T22Audio/test_features_X.npy T22Audio/train_features_X.npy y los parametros de la red neuronal se almacenan en Reconocedor/models/modelConv.yaml Reconocedor/models/modelConv.h5

## Entrenamiento de memorias asociaivas

El script test_memory.py tomará los features obtenidos en la red neuronal para entrenar las memorias, las cuales se almacenan en Reconocedor/models/AM_* y su representación gráfica en T22Audio/amsgrid_*
Este script también realiza una validación de las memorias los resultados se almacenan en T22Audio/tables.npy y T22Audio/entropies.npy, mientras que la gráfica se almacena en T22Audio/graph_T22_*
