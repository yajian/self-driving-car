# coding:utf8
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import  img_as_ubyte
from moviepy.editor import VideoFileClip

left_up=(580,460)
right_up=(740,460)
left_down=(280,680)
right_down=(1050,680)

def find_area(path='./test_images/test6.jpg'):
	image = cv2.imread(path)

	cv2.circle(image,left_up,0,(0,0,255),10)
	cv2.circle(image,right_up,0,(0,0,255),10)
	cv2.circle(image,left_down,0,(0,0,255),10)
	cv2.circle(image,right_down,0,(0,0,255),10)
	rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	plt.imshow(rgb_image)
	plt.show()

def perspective_transform(image):
	img_h = image.shape[0]
	img_w = image.shape[1]
	#原图像待变换区域顶点坐标
	src = np.float32([left_up,left_down,right_up,right_down])
	#目标区域顶点坐标
	dst = np.float32([[200,0],[200,680],[1000,0],[1000,680]])
	#求得变换矩阵
	M = cv2.getPerspectiveTransform(src,dst)
	#进行透视变换
	warped = cv2.warpPerspective(image,M,(img_w,img_h),flags=cv2.INTER_NEAREST)
	return img_as_ubyte(warped),M

def gaussian_blur(image,kernel=(5,5)):
	return cv2.GaussianBlur(image,kernel,0)

def edge_detection(image,sobel_kernel=3,sc_threshold=(110, 255), sx_threshold=(20, 100)):

	img = np.copy(image)
	#转换为hsv格式
	hsv = cv2.cvtColor(img,cv2.COLOR_RGB2HLS).astype(np.float)
	h_channel = hsv[:,:,0]
	l_channel = hsv[:,:,1]
	s_channel = hsv[:,:,2]
	#使用s通道
	channel = s_channel
	#x方向梯度
	sobel_x = cv2.Sobel(channel,cv2.CV_64F,1,0,ksize = sobel_kernel)
	#二值化处理
	scaled_sobel_x = cv2.convertScaleAbs(255*sobel_x/np.max(sobel_x))
	sx_binary = np.zeros_like(scaled_sobel_x)
	#进行边缘检测
	sx_binary[(scaled_sobel_x >= sx_threshold[0]) & (scaled_sobel_x<=sx_threshold[1])] = 1
	s_binary = np.zeros_like(s_channel)
	#进行颜色检测
	s_binary[(channel>=sc_threshold[0]) & (channel<=sc_threshold[1])]=1
	flat_binary = np.zeros_like(sx_binary)
	#颜色和边缘叠加
	flat_binary[(sx_binary == 1) | (s_binary ==1)] =1

	return flat_binary

def roi(img,points=[(120, 710),(280, 0), (1100, 0), (1050,710)]):
	mask = np.zeros_like(img)

	if len(img.shape) >2:
		channel_count = image.shape[2]
		ignore_mask_color=(255,)*channel_count
	else:
		ignore_mask_color = 255
	vertices = np.array([points],dtype=np.int32)
	cv2.fillPoly(mask,vertices,ignore_mask_color)
	masked_image = cv2.bitwise_and(img,mask)
	return masked_image

def color_compare():
	image,M = perspective_transform()
	canny_image = dege_detection(image)
	
	f,(ax1,ax2) = plt.subplots(1,2,figsize=(10,20))
	ax1.imshow(image[:,:,::-1])
	ax1.set_title('original image')
	ax2.imshow(image[:,:,0])
	ax2.set_title("blue channel")
	plt.show()

def plot_hist(image,axis=0):
	histogram = np.sum(image[image.shape[0]/2:,:],axis)
	print np.argmax(histogram)
	plt.plot(histogram)
	plt.show()

def findlines(roi_image):
	#获取直方图，得到车道大致位置
	histogram = np.sum(roi_image[int(roi_image.shape[0]/2):,:],axis=0)
	#输出图像
	out_image = np.dstack((roi_image,roi_image,roi_image)) * 255
	
	#图像中线x轴的坐标
	midpoint = np.int(histogram.shape[0]/2)
	#图像中线左侧车道x轴基准
	leftx_base = np.argmax(histogram[:midpoint])
	#图像中线右侧车道x轴基准
	rightx_base = np.argmax(histogram[midpoint:]) + midpoint

	#设置滑动窗口个数
	nwindows = 9
	#窗口高度
	window_height = np.int(roi_image.shape[0]/nwindows)
	#所有非零点坐标
	nonzero = roi_image.nonzero()
	#非零点y坐标，因为矩阵下标为（行，列），而行在坐标系里代表y轴
	nonzeroy = nonzero[0]
	#非零点x坐标,因为矩阵下标为（行，列），而列在坐标系里代表x轴
	nonzerox = nonzero[1]
	#左车道搜索起始点
	leftx_current = leftx_base
	#右车道搜索起始点
	rightx_current = rightx_base
	#窗口长度2*margin，即搜索中线左右80像素范围内的点
	margin = 80
	#窗口内超过50个点才会改变搜索中线
	minpix = 50
	#左车道窗口内像素点坐标
	left_lane_indices = []
	#右车道窗口内像素点坐标
	right_lane_indices = []


	for window in range(nwindows):
		#窗口上边线y值
		win_y_low = roi_image.shape[0] - (window + 1) * window_height
		#窗口下边线y值
		win_y_high = roi_image.shape[0] - window * window_height
		#窗口左下顶点坐标
		win_xleft_low = leftx_current - margin
		#窗口左上顶点坐标
		win_xleft_high = leftx_current + margin
		#窗口右下顶点坐标
		win_xright_low = rightx_current - margin
		#窗口右上顶点坐标
		win_xright_high = rightx_current + margin
		#画图
		# cv2.rectangle(out_image,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(255,0,0),2)
		# cv2.rectangle(out_image,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(255,0,0),2)
		#挑出左窗口范围内的点
		good_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
			& (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
		#挑出右窗口范围内的点
		good_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
			& (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

		left_lane_indices.append(good_left_indices)
		right_lane_indices.append(good_right_indices)

		if len(good_left_indices) > minpix:
			#将左车道搜索中线置为窗口内像素点x坐标的平均值
			leftx_current = np.int(np.mean(nonzerox[good_left_indices]))
		if len(good_right_indices) > minpix:
			#将右车道搜索中线置为窗口内像素点x坐标的平均值
			rightx_current = np.int(np.mean(nonzerox[good_right_indices]))
 
 	left_lane_indices = np.concatenate(left_lane_indices)
 	right_lane_indices = np.concatenate(right_lane_indices)

 	leftx = nonzerox[left_lane_indices]
 	lefty = nonzeroy[left_lane_indices]
 	rightx = nonzerox[right_lane_indices]
 	righty = nonzeroy[right_lane_indices]
 	#拟合左车道曲线
 	left_fit = np.polyfit(lefty,leftx,2)
 	#拟合右车道曲线
 	right_fit = np.polyfit(righty,rightx,2)
 	#y轴的坐标都是确定的，待确定的是x的坐标
 	ploty = np.linspace(0,roi_image.shape[0]-1,roi_image.shape[0])
 	#根据左车道y的坐标计算x的坐标
 	left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
 	#根据右车道y的坐标计算x的坐标
 	right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
 	#标记左车道点为红色
 	out_image[nonzeroy[left_lane_indices],nonzerox[left_lane_indices]] = [255,0,0]
 	#标记右车道点为蓝色
 	out_image[nonzeroy[right_lane_indices],nonzerox[right_lane_indices]] = [0,0,255]


 	ym_per_pix = 30.0/720
 	xm_per_pix = 3.7/720
 	y_eval = np.max(ploty)

 	left_fit_cr = np.polyfit(ploty * ym_per_pix,left_fitx * ym_per_pix,2)
 	right_fit_cr = np.polyfit(ploty * xm_per_pix,right_fitx * ym_per_pix,2)

 	#曲率半径公式
 	left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5)/np.absolute(2 * left_fit_cr[0])
 	right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5)/np.absolute(2 * right_fit_cr[0])

 	m_car = roi_image.shape[1]/2
 	m_lane = (left_fitx[0] + right_fitx[0])/2
 	offset_right_from_center_m = (m_lane - m_car)*xm_per_pix
 	avg_radius_meters = np.mean([left_curverad,right_curverad])

	return out_image,avg_radius_meters,offset_right_from_center_m,left_fitx,right_fitx,ploty

def to_real_world_scale(original,out_image,M,left_fitx,right_fitx,ploty):
	color_warp = np.zeros_like(out_image).astype(np.uint8)
	pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
	pts = np.hstack((pts_left,pts_right))
	cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
	newwarp = cv2.warpPerspective(color_warp,np.linalg.inv(M),(original.shape[1],original.shape[0]))
	result = cv2.addWeighted(original,1,newwarp,0.3,0)
	return result

def mark_lanes(original,out_image,M,left_fitx,right_fitx,ploty):
	color_warp = np.zeros_like(out_image).astype(np.uint8)
	pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
	pts = np.hstack((pts_left,pts_right))
	cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))
	marked = cv2.addWeighted(original,1,color_warp,0.3,0)
	return marked

old_image_lines = None
l_fit_buffer = None
r_fit_buffer = None

def process_image(image):
	transform_image,M = perspective_transform(image)
	edge_image = edge_detection(transform_image)
	roi_image = roi(edge_image)
	out_image,avg_radius_meters,offset_right_from_center_m,left_fitx,right_fitx,ploty = findlines(roi_image)

	global old_image_lines
	global l_fit_buffer
	global r_fit_buffer

	#smoothing
	if old_image_lines is None:
		old_image_lines = roi_image

	ret = cv2.matchShapes(old_image_lines,roi_image,1,0.0)
	if ret<50:
		old_image_lines = roi_image
		if l_fit_buffer is None:
			l_fit_buffer = np.array([left_fitx])

		if r_fit_buffer is None: 
			r_fit_buffer = np.array([right_fitx])

		l_fit_buffer = np.append(l_fit_buffer,[left_fitx],axis=0)[-20:]
		r_fit_buffer = np.append(r_fit_buffer,[right_fitx],axis=0)[-20:]

	l_fit_mean = np.mean(l_fit_buffer,axis=0)
	r_fit_mean = np.mean(r_fit_buffer,axis=0)
	after_process_image = to_real_world_scale(image,out_image,M,l_fit_mean,r_fit_mean,ploty)
	return after_process_image

def show_mark_lanes():
	image = cv2.imread('./test_images/test6.jpg')
	transform_image,M = perspective_transform(image)
	edge_image = edge_detection(transform_image)
	roi_image = roi(edge_image)
	out_image,avg_radius_meters,offset_right_from_center_m,left_fitx,right_fitx,ploty = findlines(roi_image)
	mark_lanes_on_bird_eye = mark_lanes(transform_image,out_image,M,left_fitx,right_fitx,ploty)
	mark_lanes_on_original = to_real_world_scale(image,out_image,M,left_fitx,right_fitx,ploty)
	f,(ax1,ax2) = plt.subplots(1,2)
	ax1.imshow(mark_lanes_on_bird_eye)
	ax1.set_title('mark lanes on bird eye pic')
	ax2.imshow(mark_lanes_on_original)
	ax2.set_title('mark lanes on original pic')
	plt.show()

def main():
	video_output = './output_video/output.mp4'
	clip1 = VideoFileClip('./original.mp4')
	white_clip = clip1.fl_image(process_image)
	white_clip.write_videofile(video_output,audio=False)

if __name__== '__main__':
	main()