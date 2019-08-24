"""
Name: Ciaran Cooney
Date: 07/05/2019
Description: Tensorflow implementation of LSTM classifier for classifying GPS coordinates.
Performance not as good as with DNN.
"""
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.utils import normalize

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
#from keras.layers.normalization import BatchNormalization # not implemented
from keras.callbacks import ReduceLROnPlateau, EarlyStopping#, ModelCheckpoint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler

def journey_time(x,y):
    """
    Compute journey time in seconds.
    """
    x = pd.to_datetime(x)
    y = pd.to_datetime(y)
    return (y-x).total_seconds()

def accuracy_f_score(y_pred,y_true):
    print(f"Accuracy score: {round(accuracy_score(y_true, y_pred) * 100,2)}%")
    print('\033[92m' + f"F1 score: {f1_score(y_true, y_pred)}" + '\033[0m')
    

def in_city(y_pred):
    targets = []
    for pred in y_pred:
        if (3750901.5068 <= pred[0] <= 3770901.5069) and (-19268905.6133 <= pred[1] <= -19208905.6133):
            targets.append(1)
        else:
            targets.append(0)
    return targets

def sigmoid(x):
	e = np.exp(1)
	y = 1/(1+e**(-x))
	return y 


df = pd.read_csv('data_train/final_locations.csv')
df["dpc"] = list(map(sigmoid, df["dist_pct_ch"]))
df.drop(["hash","vmax","vmin","vmean","time_entry","time_exit","x_home","y_home","nj","dist_pct_ch"],axis=1, inplace=True)
df.set_index("trajectory_id", inplace=True)

y = [df["x_exit"].values, df["y_exit"].values]
y = np.transpose(np.array(y))
df.drop(["x_exit","y_exit"], axis=1, inplace=True)

scaler = MinMaxScaler(feature_range=(-1,1))
X = df.values
X = scaler.fit_transform(X)


EPOCHS = 40
BATCH_SIZE = 32 

tf.set_random_seed(0)
np.random.seed(24)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

model = None
model = Sequential()

model.add(LSTM(64,activation='relu',return_sequences=False)) #False before Dense layer
model.add(Dropout(0.1))
model.add(BatchNormalization())

# model.add(LSTM(64, activation='relu',kernel_initializer='normal',return_sequences=False))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())

model.add(Dense(1024, activation='relu',kernel_initializer='normal'))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(Dense(2, activation='linear'))

optimiser = tf.keras.optimizers.Adam(lr=1.0, decay=1e-5) #lr=1e-2 = 31.82% valacc


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5' 
checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')
model.compile(loss='mean_absolute_percentage_error', optimizer=optimiser, metrics=['accuracy']) # try different loss functions
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=410,shuffle=True)


model.fit(x_train,y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[checkpoint])
