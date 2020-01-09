# imports needed for a CNN

import numpy as np
import csv
import cv2
import os, glob
import tensorflow as tf
import random
import sklearn
from sklearn.model_selection import train_test_split
import time
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from matplotlib import pyplot as plt
from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

def load_data(data_dir):
	
	# Get all subdirectories. A folder for each class
    #if os.path.isdir(os.path.join(data_dir, d))]
	directories = [d for d in os.listdir(data_dir)
					if os.path.isdir(os.path.join(data_dir, d))]
	#print(directories)
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
	labels = []
	images = []

	category = 0
	for d in directories:
		label_dir = os.path.join(data_dir, d)
		file_names = [os.path.join(label_dir, f)
					for f in os.listdir(label_dir)
					if f.endswith(".jpg")]
        # adding an early stop for sake of speed
		#stop = 0
		for f in file_names:
			img = cv2.imread(f)
			imresize = cv2.resize(img, (100, 100))
			#plt.imshow(imresize)
			images.append(imresize)
			labels.append(category)
			# remove this to use full data set
			#if stop > 30:
			#break
			#stop += 1
			#print(stop)
			# end early stop
            
		category += 1

	return images, labels

data_dir = "/home/sushmithaagowdaa96/fruit-bin/fruit360/training_images"
training_images, training_labels = load_data(data_dir)

# confirm that we have the data
print(len(training_images))
print(len(training_labels))
print("Training set dims")
#cv2.imshow(str(training_labels[random.randint(0,(len(training_labels)-1))]), training_images[random.randint(0,(len(training_images)-1))])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

data_dir = "/home/sushmithaagowdaa96/fruit-bin/fruit360/validate"
validation_images, validation_labels = load_data(data_dir)

# confirm that we have the datain
print("Validation set dims")
print(len(validation_images))
print(len(validation_labels))
#cv2.imshow(str(validation_labels[random.randint(0,(len(validation_labels)-1))]), validation_images[random.randint(0,(len(validation_images)-1))])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

data_dir = "/home/sushmithaagowdaa96/fruit-bin/fruit360/testing_images"
testing_images, testing_labels = load_data(data_dir)

# confirm that we have the data
print("Test set dims")
print(len(testing_images))
print(len(testing_labels))
#cv2.imshow(str(testing_labels[random.randint(0,(len(testing_labels)-1))]), testing_images[random.randint(0,(len(testing_images)-1))])
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# normalize inputs from 0-255 and 0.0-1.0
X_train = np.array(training_images).astype('float32')
X_cv = np.array(validation_images).astype('float32')
X_test = np.array(testing_images).astype('float32')
X_train = X_train / 255.0
X_cv = X_cv / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np.array(training_labels)
y_cv = np.array(validation_labels)
# y_test = np.array(testing_labels) %%%% Irrelevant Data
y_train = np_utils.to_categorical(y_train)
y_cv = np_utils.to_categorical(y_cv)
# y_test = np_utils.to_categorical(y_test) %%%% Irrelevant Data
num_classes = y_cv.shape[1]
# print details for debugging
print("X_train")
print(len(X_train))
print("X_cv")
print(len(X_cv))
print("y_train")
print(len(y_train))
print("y_cv")
print(len(y_cv))
print("Number of classes")
print(num_classes)
print("Data normalized and hot encoded.")

def create_cnn_model(num_classes, lrate):
	# Create model
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(100, 100, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add (Dropout(0.2))
	model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile Model
	epochs = 1
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(loss='categorical_crossentropy', optimizer=tf.train.RMSPropOptimizer(learning_rate=0.001), metrics=['accuracy'])
	print(model.summary())
	return model, epochs

# Create CNN model
model, epochs = create_cnn_model(num_classes, 0.01)
print(" CNN model created. ")

# fit and run our model
seed = 7
np.random.seed(seed)
model.fit(X_train, y_train, validation_data=(X_cv, y_cv), epochs=epochs, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_cv, y_cv, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
print("Training Done")

# Test the ConvNet

print("Evaluating test images ")
cnn_prediction = model.predict_on_batch(X_test)
print(cnn_prediction)
print("Predictions Done")
