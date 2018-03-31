#coding:utf-8
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import pickle

#摄像头校正
def camera_calibration():
	#横向角点数量
	nx = 9
	#纵向角点数量
	ny = 6
	#获取角点坐标
	objp = np.zeros((nx*ny,3),np.float32)
	#赋值x,y,默认z=0
	objp[:,:2] = np.mgrid[0:nx,0:ny].transpose(2,1,0).reshape(-1,2)

	objpoints = []
	imgpoints = []

	images = glob.glob("./camera_cal/calibration*.jpg")
	count = 0 
	plt.figure(figsize=(12,8))
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		#标记棋盘角点
		ret,corners = cv2.findChessboardCorners(gray,(nx,ny),None)

		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
			#角点图
			img_cor = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
			#画图
			plt.subplot(4,5,count+1)
			plt.axis('off')
			plt.title(fname.split('/')[-1])
			plt.imshow(img_cor)
			count += 1
			write_name = './corners_found/corners_'+fname.split('/')[-1];
			cv2.imwrite(write_name,img)
	plt.show()
	return objpoints,imgpoints

#保存校正系数等
def save_parameters(objpoints,imgpoints):
	img = cv2.imread('./camera_cal/calibration1.jpg')
	img_size = (img.shape[1],img.shape[0])
	ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
	dist_pickle = {}
	dist_pickle['mtx'] = mtx
	dist_pickle['dist'] = dist
	pickle.dump(dist_pickle,open('./output_images/camera_mtx_dist.p','wb'))
	print 'parameters saved'


def main():
	objpoints,imgpoints = camera_calibration()
	save_parameters(objpoints,imgpoints)

if __name__=='__main__':
	main()