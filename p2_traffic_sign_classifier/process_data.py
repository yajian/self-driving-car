import numpy as np
import pickle
import os
import cv2
import csv

def process_train_data(path):
	file = os.listdir(path)
	classes = len(file)
	train_data = []
	train_labels = []
	for i in range(0,classes):
		dir_name = file[i]
		if dir_name=='.DS_Store':
			continue
		full_dir_path = path + dir_name
		csv_file_path = full_dir_path + '/' + 'GT-{0}.csv'.format(dir_name)
		with open(csv_file_path) as f:
			csv_reader = csv.reader(f,delimiter=';')
			# pass header
			csv_reader.next()
			for (filename,width,height,x1,y1,x2,y2,classid) in csv_reader:
				train_labels.append(classid)
				image_file_path = full_dir_path+'/'+filename
				resized_image = resize_image(image_file_path,(x1,y1,x2,y2))
				train_data.append(resized_image)
			f.close()
	print 'train data process done'
	return train_data,train_labels

def resize_image(path,index):
	image = cv2.imread(path)
	image = image[int(index[0]):int(index[2]),int(index[1]):int(index[3])]
	res = cv2.resize(image,(32,32),interpolation = cv2.INTER_CUBIC)
	return res 

def process_test_data(path):
	test_data = []
	test_labels = []
	csv_file_path = path + '/' + 'GT-final_test.csv'
	with open(csv_file_path) as f:
		csv_reader = csv.reader(f,delimiter=';')
		csv_reader.next()
		for (filename,width,height,x1,y1,x2,y2,classid) in csv_reader:
			test_labels.append(classid)
			image_file_path = path+'/'+filename
			resized_image = resize_image(image_file_path,(x1,y1,x2,y2))
			test_data.append(resized_image)
	print 'test data process done'
	return test_data,test_labels

def main():
	train_data_path = '../data/GTSRB/Final_Training/Images/'
	test_data_path = '../data/GTSRB2/Final_Test/Images'
	train_data,train_labels = process_train_data(train_data_path)
	test_data,test_labels = process_test_data(test_data_path)
	with open('./train.p','wb') as f:
		pickle.dump({"data":np.array(train_data),"labels":np.array(train_labels)},f)
	with open('./test.p','wb') as f:
		pickle.dump({"data":np.array(test_data),"labels":np.array(test_labels)},f)

if __name__=="__main__":
	main()
