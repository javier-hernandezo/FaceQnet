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
	
	for imagen in [imagen for imagen in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, imagen))]:
		imagenes = os.path.join(image_path, imagen)
		print(imagenes)
		img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (224, 224))
		X_test.append(img)
		images_names.append(imagenes)
	return X_test, images_names

def read_and_normalize_test_data():
    test_data, images_names = load_test()
    test_data = np.array(test_data, copy=False, dtype=np.float32)
    return test_data, images_names
		

#Loading one of the the pretrained models

# model = load_model('FaceQnet.h5')

model = load_model('FaceQnet_v1.h5')

#See the details (layers) of FaceQnet
# print(model.summary())

#Loading the test data
test_data, images_names = read_and_normalize_test_data()
y=test_data

#Extract quality scores for the samples
score = model.predict(y, batch_size=batch_size, verbose=1)
predictions = score ;


#Saving the quality measures for the test images
fichero_scores = open('scores_quality.txt','w')
i=0


#Saving the scores in a file
fichero_scores.write("img;score\n")
for item in predictions:
	fichero_scores.write("%s" % images_names[i])
	#Constraining the output scores to the 0-1 range
	#0 means worst quality, 1 means best quality
	if float(predictions[i])<0:
		predictions[i]='0'
	elif float(predictions[i])>1:
		predictions[i]='1'
	fichero_scores.write(";%s\n" % predictions[i])
	i=i+1
