from keras.datasets import mnist
import sys
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
	# compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def MLP_model():
    model = Sequential()
    #Data is in format 28*28  single dimensional
    model.add(Dense(79, activation='relu', kernel_initializer='he_uniform',input_dim=784))
    model.add(Dense(10, activation='softmax'))
	# compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def summarize_diagnostics(history,filename):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	pyplot.tight_layout()
	# save plot to file
	pyplot.savefig('images/Q3_'+filename+'.png')
	pyplot.close()

def run_test_harness(model,filename,train_it,test_it):
	# fit model
	history = model.fit(train_it[0],train_it[1], steps_per_epoch=len(train_it),
		validation_data=(test_it[0],test_it[1]), validation_steps=len(test_it), epochs=20, verbose=0)
	# evaluate model
	_, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
	print(filename+'> %.3f' % (acc * 100.0))
	# learning curves
	summarize_diagnostics(history,filename)

#loading
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X=train_X.reshape(train_X.shape[0],train_X.shape[1],train_X.shape[2])
test_X=test_X.reshape(test_X.shape[0],test_X.shape[1],test_X.shape[2])
train_y=to_categorical(train_y,10)
test_y=to_categorical(test_y,10)


train_it=[train_X,train_y]
test_it=[test_X,test_y]

print("For CNN\n")
cnn=CNN_model()
print(cnn.summary())
run_test_harness(cnn,"CNN",train_it,test_it)

#Reshaping to 1D as dense in mlp takes 1 d input
train_X=train_X.reshape(train_X.shape[0],train_X.shape[1]*train_X.shape[2])
test_X=test_X.reshape(test_X.shape[0],test_X.shape[1]*test_X.shape[2])
train_it=[train_X,train_y]
test_it=[test_X,test_y]
print("For MLP\n")
mlp=MLP_model()
print(mlp.summary())
run_test_harness(mlp,"MLP",train_it,test_it)


# #shape of dataset
# print('X_train: ' + str(train_X.shape))
# print('Y_train: ' + str(train_y.shape))
# print('X_test:  '  + str(test_X.shape))
# print('Y_test:  '  + str(test_y.shape))

# #plotting
# for i in range(9):  
#     pyplot.subplot(330 + 1 + i)
#     pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
#     pyplot.title(str(train_y[i]))
# pyplot.show()