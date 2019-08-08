'''
Function to train driver distraction detection
date: 22.07.2019
Arguments:  -m --> to train mobile only network
            -t --> to test a trained network and generate confusion matrix
            -p --> to provide a new path of the trained model. Default pathn is 
                    same folder
'''

import keras
import sys
import matplotlib.pyplot as plt
import model as m
import loader as L
import numpy
from sklearn.metrics import confusion_matrix

from keras_preprocessing.image import ImageDataGenerator

suffix = None

if '-m' in sys.argv:
    #-m for Only mobile
    print("INFO: Training for mobile phone related only distractions ...........")
    model = m.model()[0]
    suffix = "Mobile"
else:
    print("INFO: Training for all distractions ...........")
    model = m.model()[0]
    suffix= "All"

# Number of epochs 
epoch = 50 # 500 for augmentation
batch_size = 32 #32 used by authors

x_train, y_train, x_val, y_val = L.getArrays()
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
x_val = numpy.array(x_val)
y_val = numpy.array(y_val)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
datagen.fit(x_train)

validation_generator = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
validation_generator.fit(x_val)


# To stop if sufficietly trained
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model' + suffix + '.h5', monitor='val_loss', mode='min')

cb = [mc]

rmsprop = keras.optimizers.RMSprop(lr=0.00001, decay=1e-5) #rho=0.9, epsilon=None
adam = keras.optimizers.adam(lr=0.0001, decay=1e-5) #rho=0.9, epsilon=None

model.compile(optimizer=rmsprop, loss='mean_squared_error', metrics=['accuracy']) # categorical_crossentropy

# To check the validity of data
for ele in x_train:
    if numpy.isnan(numpy.sum(ele)):
        print("Found NAAAAAAAAAAAAAAAAAAAAAAAAAAAAN")
    if numpy.isinf(numpy.sum(ele)):
        print("Found INFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
for ele in x_val:
    if numpy.isnan(numpy.sum(ele)):
        print("Found NAAAAAAAAAAAAAAAAAAAAAAAAAAAAN")
    if numpy.isinf(numpy.sum(ele)):
        print("Found INFFFFFFFFFFFFFFFFFFFFFFFFFFFF")

def getTrainedModel(p='best_modelAll.h5'):
    if ('-m' in sys.argv) and not('-p' in sys.argv):
        p = 'best_modelMobile.h5'
    if '-p' in sys.argv:
        for arg in sys.argv:
            if not(arg in ["main.py", "-p", "-m", "-t"]):
                p = arg
    return(keras.models.load_model(p))

def plotSaveConfusionMatrix(matrix, classes="All"):
    #for matrix in confusion_matrices:
    fig, ax = plt.subplots(figsize=(8,8))
    ax.matshow(matrix, cmap='seismic', aspect='auto')
    for (i, j), z in numpy.ndenumerate(matrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center',fontsize=8, color='white')
    plt.title('Confusion Matrix DDD : ' + classes + ' Classes', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    if classes == "All":
        l = numpy.array([0,1,2,3,4,5,6,7,8,9])
        c = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    else:
        l = numpy.array([0,1,2,3,4])
        c = ['c0','c1','c2','c3','c4']
    plt.xticks(l, c)
    plt.yticks(l, c)
    plt.savefig(classes + 'Classes' +'Confusion_matrix'+'.png')

if '-t' in sys.argv:
    #Testing previously trained model
    if '-t' in sys.argv:
        model = getTrainedModel()
    score = model.evaluate(x_val, y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    predictions = model.predict(x_val)
    matrix = confusion_matrix(y_val.argmax(axis=1), predictions.argmax(axis=1))
    if '-m'in sys.argv:
        classes = 'Mobile'
    else:
        classes = 'All'
    print("Confusion Matrix : ", matrix)
    plotSaveConfusionMatrix(matrix, classes)
else:
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, batch_size=batch_size, callbacks=cb)
    #history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), validation_data=(x_val, y_val), steps_per_epoch=len(x_train) / batch_size, epochs=epoch, callbacks=cb)

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Accuracy" + suffix + ".png")
    plt.show()

    # Plot training & validation loss values
    plt.clf() # Clear earlier figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("Loss" + suffix + ".png")
    plt.show()