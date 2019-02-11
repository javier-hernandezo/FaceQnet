# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os
import cv2


batch_size = 1 #Numero de muestras para cada batch (grupo de entrada)

def load_test():
	X_test = []
	images_names = []
	image_path = r".\\Samples_cropped\\"
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
model = load_model('FaceQnet.h5')

#See the details (layers) of FaceQnet
# print(model.summary())

#Loading the test data
test_data, images_names = read_and_normalize_test_data()
# if test_data.ndim == 4:
	# axis = (1, 2, 3)
	# size = test_data[0].size
# elif test_data.ndim == 3:
	# axis = (0, 1, 2)
	# size = test_data.size
# else:
	# raise ValueError('Dimension should be 3 or 4')

# mean = np.mean(test_data, axis=axis, keepdims=True)
# std = np.std(test_data, axis=axis, keepdims=True)
# std_adj = np.maximum(std, 1.0/np.sqrt(size))
# y = (test_data - mean) / std_adj

y=test_data

#Extract quality scores for the samples
m=0.7
s=0.5
score = model.predict(y, batch_size=batch_size, verbose=1)
score = 0.5*np.tanh(((score-m)/s) + 1)
predictions = -score + 1; #Convertimos el score de distancia en similitud

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
