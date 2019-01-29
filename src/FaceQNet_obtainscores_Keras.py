# -*- coding: utf-8 -*-
import keras as keras
from keras.models import Model,Sequential,load_model
import keras.layers as layers
from keras.layers import Dense
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adadelta
from keras.utils import plot_model

import numpy as np
import glob
import sys
from PIL import Image
from matplotlib import pyplot as plt
from IPython.display import clear_output
import os
import cv2
import pandas as pd


batch_size = 1 #Numero de muestras para cada batch (grupo de entrada)

def load_test():
	X_test = []
	images_names = []
	image_path = r".\\Samples\\"
	print('Read test images')
	# for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
		# carpeta= os.path.join(image_path, f)
		# print(carpeta)
		# for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
			# imagenes = os.path.join(carpeta, imagen)
			# img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
			# X_test.append(img)
			# images_names.append(imagenes)
			
	for imagen in [imagen for imagen in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, imagen))]:
		imagenes = os.path.join(image_path, imagen)
		img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
		X_test.append(img)
		images_names.append(imagenes)
	return X_test, images_names

def read_and_normalize_test_data():
    test_data, images_names = load_test()
    test_data = np.array(test_data, copy=False, dtype=np.float32)
    return test_data, images_names
		

#Loading the pretrained model
model = load_model('VGG2-Quality_NET.h5')

#See the details (layers) of FaceQnet
# print(model.summary())

#Loading the test data
test_data, images_names = read_and_normalize_test_data()

#Extract quality scores for the samples
predictions = model.predict(test_data, batch_size=batch_size, verbose=1)

#Guardamos los scores para cada clase en la prediccion de scores
fichero_scores = open('scores_quality.txt','w')
i=0

#Saving the scores in a file
fichero_scores.write("img;score\n")
for item in predictions:
	fichero_scores.write("%s" % images_names[i]) #fichero
	#Contraining the output scores to the 0-1 range
	#0 means worst quality, 1 means best quality
	if float(predictions[i])<0:
		predictions[i]='0'
	elif float(predictions[i])>1:
		predictions[i]='1'
	fichero_scores.write(";%s\n" % predictions[i])
	i=i+1
