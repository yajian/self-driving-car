#coding:utf-8
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
#读取校正系数
def load_undistort_parameter(path = './output_images/camera_mtx_dist.p'):
	with open(path) as f:
		data = pickle.load(f)
		return data['mtx'],data['dist']
#消除畸变
def undistort(img,mtx,dist):
	return cv2.undistort(img,mtx,dist,None,mtx)
#展示效果图
def show_undistorted_image(mtx,dist,path = './camera_cal/calibration1.jpg'):
	img = cv2.imread(path)
	undst = undistort(img,mtx,dist)
	#为了显示把BGR格式转为RGB格式
	img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	undst = cv2.cvtColor(undst,cv2.COLOR_BGR2RGB)
	plt.figure(figsize=(200,200))
	plt.subplot(1,2,1)
	plt.title('origin image')
	plt.imshow(img)
	plt.subplot(1,2,2)
	plt.title('undistort image')
	plt.imshow(undst)
	plt.show()

def main():
	mtx,dist = load_undistort_parameter()
	show_undistorted_image(mtx,dist)
	show_undistorted_image(mtx,dist,'./test_images/test1.jpg')

if __name__=='__main__':
	main()