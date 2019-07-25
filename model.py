'''
Cnn model described in https:\\doi.org/10.1007/978-3-319-68612-7_9
But used for all classes (C0 to C9)
date: 22.07.2019
Arguments: -s --> to save the model as "model.png"
           -m --> for only for mobile classes (C0 to C4)
default : all classes (C0 to C9)
'''

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import plot_model
import sys


def model():
    #optional argument m for mobile model

    #create model
    model = Sequential()

    #adding model layers

    #First layer: 2D convolution (32,3,3)
    model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(150,150,3)))

    #Second layer: 2D max pooling (2,2)
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')) # data_format = channels_last or data_format = channels_first

    #Third layer: 2D convolution (32,3,3)
    model.add(Conv2D(32, kernel_size=3, activation="relu")) #input_shape=(150,150,3)

    #Fourth layer: 2D max pooling (2,2)
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')) # data_format = channels_last or data_format = channels_first

    #Fifth layer: 2D convolution (64,3,3)
    model.add(Conv2D(64, kernel_size=3, activation="relu")) #input_shape=(150,150,3)

    #Sixth layer: 2D max pooling (2,2)
    model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid')) # data_format = channels_last or data_format = channels_first

    #Final output layer
    model.add(Flatten())
    if '-m' in sys.argv:
        f = True
        model.add(Dense(5, activation="sigmoid"))
    else:
        f = False
        model.add(Dense(10, activation="sigmoid"))
    
    return(model, f)

if '-s' in sys.argv:
    #Save model image
    m, f = model()
    n = 'model.png'
    if f:
        n = 'model_onlyMobile.png'
    plot_model(m, to_file=n, show_shapes=True)

