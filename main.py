'''
Function to train driver distraction detection
date: 22.07.2019
Arguments: -m --> to train mobile only network
'''

import keras
import sys
import matplotlib.pyplot as plt
import model as m
import loader as L
import numpy

#model = None

if '-m' in sys.argv:
    #-m for Only mobile
    print("INFO: Training for mobile phone related only distractions ...........")
    model = m.model()[0]
else:
	print("INFO: Training for all distractions ...........")
	model = m.model()[0]

# Number of epochs 
epoch = 10
batch_size = 200

x_train, y_train, x_val, y_val = L.getArrays()
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
x_val = numpy.array(x_val)
y_val = numpy.array(y_val)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, batch_size=batch_size)

print(history.history)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()