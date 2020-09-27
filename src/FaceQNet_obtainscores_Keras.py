# -*- coding: utf-8 -*-

from keras.models import load_model
import numpy as np
import os
import cv2


batch_size = 1 #Number of samples for each batch (input group)
images_dir = r"./Samples_cropped/"

def load_test_data():
	X_test = []
	images_names = []
	print('Reading images: ')
	for image_name in os.listdir(images_dir):
		image_path = os.path.join(images_dir, image_name)
		if image_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
			print(image_path)
			image = cv2.resize(cv2.imread(image_path, cv2.IMREAD_COLOR), (224, 224))
			X_test.append(image)
			images_names.append(image_path)
	return X_test, images_names

def read_and_normalize_test_data():
    X_test, images_names = load_test_data()
    X_test = np.array(X_test, copy=False, dtype=np.float32)
    return X_test, images_names

#Loading one of the the pretrained models
# model = load_model('FaceQnet.h5')
model = load_model('FaceQnet_v1.h5')

#See the details (layers) of FaceQnet
# print(model.summary())

#Loading the test images
X_test, images_names = read_and_normalize_test_data()

#Calculate quality scores for test images
scores = model.predict(X_test, batch_size=batch_size, verbose=1)

image_score_dict = dict(zip(images_names, scores))
image_score_list = sorted(image_score_dict.items(), key=lambda x: x[1], reverse=True)
image_score_array = np.array(image_score_list)
np.savetxt(images_dir + "face_quality_scores.csv", image_score_array, fmt='%s', delimiter=',')
