from keras.models import load_model
import os
import cv2
import matplotlib.pyplot as plt
import PIL
import csv
import numpy as np

input_dir = './test_images'
output_dir = './resize_image'

def process_test_data():
	test_images = os.listdir(input_dir)
	for image_name in test_images:
		image_path = input_dir + '/' + image_name
		image = cv2.imread(image_path)
		resize_image = cv2.resize(image,(32,32),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(output_dir + '/' + image_name,resize_image)

def load_test_data():
	test_data = []
	resize_images = os.listdir(output_dir)
	print resize_images
	for image_name in resize_images:
		image_path = output_dir + '/' + image_name
		image = cv2.imread(image_path)
		test_data.append(image)
	return test_data

def load_label():
	file_path = './signnames.csv'
	signnames = []
	with open(file_path) as f:
		lines = csv.reader(f,delimiter=',')
		lines.next()
		for line in lines:
			signnames.append(line[1])
	return signnames


def main():
	process_test_data()
	plt.figure(num='test_data',figsize=(32,32)) 
	model = load_model('./model.h5')
	test_data = load_test_data()
	predict = model.predict(np.array(test_data))
	signnames = load_label()

	for i in range(9):
		image = test_data[i]
		plt.subplot(2,9,i+1)
		plt.title("No." + str(i+1) +" image")
		plt.imshow(image)

	for i in range(9):
		image = test_data[i]
		index = predict[i]
		res = np.argmax(index,axis=0)
		title = signnames[res]
		plt.subplot(2,9,9+i+1)
		plt.title(title, fontsize=10)
		plt.imshow(image)
	plt.show()

if __name__ == '__main__':
	main()
