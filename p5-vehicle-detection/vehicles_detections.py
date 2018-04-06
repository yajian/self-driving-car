#encoding:utf8
import cv2
import numpy as np
from hog_feature import load_car_data,load_non_car_data,get_hog_feature
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import pickle
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip


def extract_features(imgs,cspace,orient=11,pix_per_cell=16,cell_per_block=2,hog_channel="ALL"):
	features=[]
	for file in imgs:
		image = plt.imread(file)
		if cspace!='RGB':
			if cspace != 'RGB':
				if cspace == 'HSV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
				elif cspace == 'LUV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
				elif cspace == 'HLS':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
				elif cspace == 'YUV':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
				elif cspace == 'YCrCb':
					feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(image)
		hog_features = []
		if hog_channel == "ALL":
			for channel_num in range(feature_image.shape[2]):
				hog_feature = get_hog_feature(feature_image[:,:,channel_num],orient,
					pix_per_cell,cell_per_block,vis=False,feature_vec=True)
				hog_features.append(hog_feature)
			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hog_feature(feature_image[:,:,hog_channel],orient,
				pix_per_cell,cell_per_block,vis=False,feature_vec=True)
		features.append(hog_features)
	return features

def split_data():
	car_data = load_car_data()
	car_features = extract_features(car_data,'YUV')
	non_car_data = load_non_car_data()
	non_car_features = extract_features(non_car_data,'YUV')
	x = np.vstack((car_features,non_car_features)).astype(np.float64)
	y = np.hstack((np.ones(len(car_features)),np.zeros(len(non_car_features))))
	rand_state = np.random.randint(0,100)
	X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=rand_state)
	with open('./train_data.p',"wb") as f:
		pickle.dump({"X_train":X_train,"y_train":y_train},f)
	with open('./test_data.p',"wb") as f:
		pickle.dump({"X_test":X_test,"y_test":y_test},f)
	return X_train,X_test,y_train,y_test

def train_model(X_train,y_train):
	svc = LinearSVC()
	svc.fit(X_train,y_train)
	joblib.dump(svc,"./svc.m")
	return svc

def load_model(path='./svc.m'):
	svc = joblib.load(path)
	return svc

def mode_performance_test():
	with open('./test_data.p') as f:
		test_data = pickle.load(f)
	X_test = test_data['X_test']
	y_test = test_data['y_test']
	svc = joblib.load('./svc.m')
	predict_res = svc.predict(X_test[0:10])
	print predict_res
	print y_test[0:10]

def find_cars(img,ystart,ystop,scale,cspace,hog_channel,svc, orient, pix_per_cell, cell_per_block,show_all_rectangle=False):
	rectangles= []
	img = img.astype(np.float32)/255
	img_tosearch = img[ystart:ystop,:,:]
	if cspace != 'RGB':
		if cspace == 'HSV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
		elif cspace == 'LUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
		elif cspace == 'HLS':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
		elif cspace == 'YUV':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
		elif cspace == 'YCrCb':
			ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
	else: ctrans_tosearch = np.copy(image) 

	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    # select colorspace channel for HOG 
	if hog_channel == 'ALL':
		ch1 = ctrans_tosearch[:,:,0]
		ch2 = ctrans_tosearch[:,:,1]
		ch3 = ctrans_tosearch[:,:,2]
	else: 
		ch1 = ctrans_tosearch[:,:,hog_channel]

	#x方向上block个数
	nxblock = (ch1.shape[1]//pix_per_cell) + 1
	#y方向上block个数
	nyblock = (ch1.shape[0]//pix_per_cell) + 1
	#定义window大小
	window = 64
	#每个window的cell个数，为了后续计算，剪掉1
	nblocks_per_window = (window//pix_per_cell) - 1
	#步长
	cells_per_step = 2
	#x方向上需要移动的次数
	nxstep = (nxblock - nblocks_per_window)//cells_per_step
	#y方向上需要移动的次数
	nystep = (nyblock - nblocks_per_window)//cells_per_step

	#获取特征
	hog1 = get_hog_feature(ch1,orient,pix_per_cell,cell_per_block, feature_vec = False)
	if hog_channel == 'ALL':
		hog2 = get_hog_feature(ch2,orient,pix_per_cell,cell_per_block, feature_vec = False)
		hog3 = get_hog_feature(ch3,orient,pix_per_cell,cell_per_block, feature_vec = False)
	#滑动抽取每个窗口特征
	for xb in range(nxstep):
		for yb in range(nystep):
			ypos = yb * cells_per_step
			xpos = xb * cells_per_step
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
			if hog_channel == 'ALL':
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window,xpos:xpos+nblocks_per_window].ravel()
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
			else : hog_features = hog_feat1

			xleft = xpos * pix_per_cell
			ytop = ypos * pix_per_cell
			hog_features = hog_features.reshape(1,-1)
			test_prediction = svc.predict(hog_features)
			#若是车辆
			if test_prediction == 1 or show_all_rectangle:
				xbox_left = np.int(xleft * scale)
				ytop = np.int(ytop * scale)
				win = np.int(window * scale)
				#保存窗口坐标
				rectangles.append(((xbox_left,ytop + ystart),(xbox_left + window,ytop+win+ystart)))
	return rectangles

def test_find_car(path):
	test_img = plt.imread(path)
	ystart = 400
	ystop = 656
	scale = 1.5
	colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	svc = joblib.load('./svc.m')
	rectangles = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,False)
	print(len(rectangles), 'rectangles found in image')
	return rectangles

def draw_boxes(path,boxes,color=(0,0,255),thick=6):
	img = plt.imread(path)
	for box in boxes:
		print box
		color = (np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255))
		cv2.rectangle(img,box[0],box[1],color,thick)
	plt.figure(figsize=(10,10))
	plt.imshow(img)
	plt.show()

def draw_all_boxes(path):
	test_img = plt.imread(path)
	ystart = 400
	ystop = 464
	scale = 1.0
	colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	svc = joblib.load('./svc.m')
	rectangles = find_cars(test_img, ystart, ystop, scale, colorspace, hog_channel, svc, orient, pix_per_cell, cell_per_block,True)
	for box in rectangles:
		color = (np.random.randint(0,255),np.random.randint(0,255), np.random.randint(0,255))
		cv2.rectangle(test_img,box[0],box[1],color,2)
	plt.figure(figsize=(10,10))
	plt.imshow(test_img)
	plt.show()

def add_heat(heatmap,bbox_list):
	for box in bbox_list:
		heatmap[box[0][1]:box[1][1],box[0][0]:box[1][0]] += 1
	return heatmap

def apply_threshold(heatmap,threshold):
	heatmap[heatmap<=threshold] = 0
	return heatmap

def draw_labeled_boxes(img,labels):
	rects = []
	for car_number in range(1,labels[1]+1):
		nonzeros = (labels[0] == car_number).nonzero()
		nonzeroy = np.array(nonzeros[0])
		nonzerox = np.array(nonzeros[1])
		bbox = (np.min(nonzerox),np.min(nonzeroy)),(np.max(nonzerox),np.max(nonzeroy))
		rects.append(bbox)
		cv2.rectangle(img,bbox[0],bbox[1],(0,0,255),6)
	return img,rects

svc = joblib.load('./svc.m')
def process_image(img):
	rectangles = []

	colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 11
	pix_per_cell = 16
	cell_per_block = 2
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"

	ystart = 400
	ystop = 464
	scale = 1.0
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 416
	ystop = 480
	scale = 1.0
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 400
	ystop = 496
	scale = 1.5
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 432
	ystop = 528
	scale = 1.5
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 400
	ystop = 528
	scale = 2.0
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 432
	ystop = 560
	scale = 2.0
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 400
	ystop = 596
	scale = 3.5
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))
	ystart = 464
	ystop = 660
	scale = 3.5
	rectangles.append(find_cars(img, ystart, ystop, scale, colorspace, hog_channel, svc, 
                           orient, pix_per_cell, cell_per_block))

	rectangles = [item for sublist in rectangles for item in sublist] 
    
	heatmap_img = np.zeros_like(img[:,:,0])
	fig,(ax1,ax2) = plt.subplots(1,2)
	heatmap_img = add_heat(heatmap_img, rectangles)
	heatmap_img = apply_threshold(heatmap_img, 1)
	labels = label(heatmap_img)
	draw_img, rects = draw_labeled_boxes(np.copy(img), labels)
	return draw_img


def main():
	test_out_file = './video/project_video_out.mp4'
	clip_test = VideoFileClip('./video/project_video.mp4')
	clip_test_out = clip_test.fl_image(process_image)
	clip_test_out.write_videofile(test_out_file,audio=False)



if __name__=='__main__':
	main()