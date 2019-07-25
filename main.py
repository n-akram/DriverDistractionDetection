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
epoch = 13
batch_size = 128 #32 used by authors

x_train, y_train, x_val, y_val = L.getArrays()
x_train = numpy.array(x_train)
y_train = numpy.array(y_train)
x_val = numpy.array(x_val)
y_val = numpy.array(y_val)

# To stop if sufficietly trained
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1)
mc = keras.callbacks.ModelCheckpoint('best_model' + suffix + '.h5', monitor='val_loss', mode='min')

model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['accuracy'])

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

history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epoch, batch_size=batch_size, callbacks=[es, mc])

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