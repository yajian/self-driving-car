#encoding=utf8
import cv2
import numpy as np
from skimage.feature import hog
import glob
import matplotlib.pyplot as plt

#读取car data
def load_car_data():
	vehicles = glob.glob('./vehicles/*/*.png')
	return vehicles
#读取non car data
def load_non_car_data():
	non_vehicles = glob.glob('./non-vehicles/*/*.png')
	return non_vehicles

#提取HOG特征
def get_hog_feature(img,orient,pix_per_cell,cell_per_block,vis=False,feature_vec = True):
	if vis == True:
		features,hog_image = hog(img,orientations = orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
			cells_per_block = (cell_per_block,cell_per_block),transform_sqrt=False,visualise=vis,feature_vector=feature_vec)
		return features,hog_image
	else :
		features = hog(img,orientations = orient,pixels_per_cell=(pix_per_cell,pix_per_cell),
			cells_per_block = (cell_per_block,cell_per_block),transform_sqrt=False,visualise=vis,feature_vector=feature_vec)
		return features
#显示对比
def show_hog():
	vehicles = load_car_data()
	non_vehicles = load_non_car_data()
	vehicle_img = plt.imread(vehicles[5])
	non_vehicle_img = plt.imread(non_vehicles[5])
	_,vehicle_hog_image = get_hog_feature(vehicle_img[:,:,2],9,8,8,vis=True,feature_vec=True)
	_,non_vehicle_hog_image = get_hog_feature(non_vehicle_img[:,:,2],9,8,8,vis=True,feature_vec=True)
	f,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(7,7))
	ax1.imshow(vehicle_img)
	ax1.set_title('car')
	ax2.imshow(vehicle_hog_image)
	ax2.set_title('car hog feature')
	ax3.imshow(non_vehicle_img)
	ax3.set_title('non car')
	ax4.imshow(non_vehicle_hog_image)
	ax4.set_title('non car hog feature')
	plt.show()