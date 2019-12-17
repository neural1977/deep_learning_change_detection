# -*- coding: utf-8 -*-
"""
Created on Thu May 16 17:54:32 2019

@author: gm.mommi
"""


# USAGE
# python train_network.py --dataset images --model santa_not_santa.model

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("TkAgg")

# import the necessary packages
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from keras.utils import np_utils
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
import pdb 
from keras import backend as K
import pickle
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras import models
from keras import layers
from keras import optimizers

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
#ap.add_argument("-p", "--plot", type=str, default="plot4.png",
	#help="path to output loss/accuracy plot")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to output label binarizer")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initia learning rate,
# and batch size
EPOCHS = 200
INIT_LR = 1e-3
BS = 32



# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
# load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (75, 75))
    image = img_to_array(image, data_format = "channels_last")
    #image = np.expand_dims(image, axis=0)
    data.append(image)
    # extract the class label from the image path and update the
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
#pdb.set_trace()
# scale the raw pixel intensities to the range [0, 1]
data = np.array(data)/ 255.0
#pdb.set_trace()
labels = np.array(labels)


# binarize the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)
	
	

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")

  
# Show a summary of the model. Check the number of trainable parameters
incrsv2 = InceptionResNetV2(weights= 'imagenet', input_shape=(75,75,3), classes=10, include_top= False)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model = models.Sequential() 

# Add the vgg convolutional base model
model.add(incrsv2)
 
# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
print(model.summary())



model.compile(loss="categorical_crossentropy", optimizer=opt,
          metrics=[ "accuracy"])


#Checkpoint
##### CONTROLLARE
tag = 'InceptionResNetV2_'+"{epoch:02d}-{val_acc:.2f}_"
filepath='bestmodel' + tag + '.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
csv_logger = CSVLogger('History_' + tag + '.log')

		  
# train the network
print("[INFO] training network...")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=2, callbacks=[csv_logger, checkpoint])		


'''
# early stopping 
earlyStopping = EarlyStopping(monitor='val_acc',
                              min_delta=0.005,
                              patience=5,
                              verbose=1, mode='auto', restore_best_weights=True)
						  
'''	


# save the model to disk
print("[INFO] serializing network...")
model.save( args["model"]+'_'+str(EPOCHS))

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(args["labelbin"], "wb")
f.write(pickle.dumps(lb))
f.close()


# plot the training loss and accuracy
plt.style.use("ggplot")
#plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig("C:/Users/Rosa/TESI/loss.png")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")
plt.savefig("C:/Users/Rosa/TESI/accuracy.png")
print(model.summary())