import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.activations import relu
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

def get_cnn_model(height, width, depth, nt):
    model = Sequential()
    #In the paper: "We first use 3D CNN layers to extract the spatial information
    #the size of the convolutional kernel is 1x3x3
    model.add(layers.Conv3D(32,(1,3,3),activation='relu',input_shape=(1, height,width,depth)))
    #We stack one CNN layer with kernel stride 1x1x1
    model.add(layers.Conv3D(64,(1,1,1),activation='relu'))
    #and another with stride 1x2x2
    model.add(layers.Conv3D(128,(1,2,2),activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.RepeatVector(1)) #reshaped into Nt one-dimensional vectors
    model.add(layers.LSTM(512))   # and fed into the LSTM layer to extract temporal information
    model.add(layers.Dense(512, activation='relu'))  #In the LSTM layer and the two FC layers, we use 512, 512, and 5 units, respectively
    model.add(layers.Dense(5, activation = 'linear'))

    model.compile(optimizer='rmsprop', loss = 'mse') #the training optimizer is RMSprop 
    return model
