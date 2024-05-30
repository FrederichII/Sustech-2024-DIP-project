import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
import median
from scipy.signal import medfilt
from scipy.signal import find_peaks
import os
Blue = 138
Green = 63
Red = 23
THRESHOLD = 60
LOW_THRESHOLD = 100
ANGLE = -45
MIN_AREA = 2000  
LICENSE_WIDTH = 440
LICENSE_HIGH = 140


def cv_show(name,img):
    cv2.namedWindow(name,0)
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

MAX_WIDTH = 640
alpha = 1.5
beta = 50


def locate(imgpath, debugMode = False):
    img = cv2.imread(imgpath)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    avg_brightness = np.mean(v)
    print("图像的平均亮度为：",avg_brightness)
    cv_show('img',img)
    img_hight, img_width = img.shape[:2]

#-------------------------------预处理--------------------------------------
#对图像进行缩放处理
    if img_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / img_width
        img = cv2.resize(img,(MAX_WIDTH,int(img_hight * resize_rate)),interpolation=cv2.INTER_AREA)
# cv_show('img_resize',img)

# 高斯平滑
    img_aussian = cv2.GaussianBlur(img,(5,5),1)
    if(debugMode):
        cv_show('img_aussian',img_aussian)

#中值滤波
    img_median = cv2.medianBlur(img_aussian,3)
    if(debugMode):
        cv_show('img_median',img_median)
# print('width：',img_median.shape[:2][0])
# print('h',img_median.shape[:2][1])
#------------------------------车牌定位-------------------------------------
#分离通道
    img_B = cv2.split(img_median)[0]
    img_G = cv2.split(img_median)[1]
    img_R = cv2.split(img_median)[2]
    if(debugMode):
        cv_show('img_B',img_B)
        cv_show('img_G',img_G)
        cv_show('img_R',img_R)
    
    
    img_B = cv2.split(img_median)[0]
    img_G = cv2.split(img_median)[1]
    img_R = cv2.split(img_median)[2]
    hist = cv2.calcHist([img_B], [0], None, [256], [0, 256])
    hist_median = median.median(hist,7)
    peaks,_ = find_peaks(hist_median.flatten())
     
    red_threashold = 100
    print(peaks)
    if(debugMode):
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show() 
        
        
        plt.figure()
        plt.title("Median-filtered Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist_median)
        plt.xlim([0, 256])
        plt.show()
        
    img_median = cv2.convertScaleAbs(img_median, alpha=alpha, beta=beta)
  
    for i in range(img_median.shape[:2][0]):
        for j in range(img_median.shape[:2][1]):
            if(img_B[i,j]<LOW_THRESHOLD):
                img_median[i,j,0]=0
               
   
    
    found = False
    idx = -3
    while(found == False):
        right_peak = peaks[idx]
        blue_threashold = 255 - right_peak + 10
        idx -= 1
        for i in range(img_median.shape[:2][0]):
            for j in range(img_median.shape[:2][1]):
                if  abs(img_B[i,j] - 255) < blue_threashold and abs(img_G[i,j] - 255)  > blue_threashold and img_R[i,j] < red_threashold:
                    img_median[i,j,0] = 255
                    img_median[i,j,1] = 255
                    img_median[i,j,2] = 255
                else: 
                    img_median[i,j,0] = 0
                    img_median[i,j,1] = 0
                    img_median[i,j,2] = 0
        if(debugMode):
            cv_show('img_median',img_median)

        kernel = np.ones((5,5),np.uint8)
        img_dilate = cv2.dilate(img_median,kernel,iterations = 5) #膨胀操作
        img_erosion = cv2.erode(img_dilate,kernel,iterations = 5) #腐蚀操作
        if(debugMode):
            cv_show('img_erosion',img_erosion)
        img1 = cv2.cvtColor(img_erosion,cv2.COLOR_RGB2GRAY)
        if(debugMode):
            cv_show('img1',img1)
        contours, hierarchy = cv2.findContours(img1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for i in range(len(contours)):
    #     print(contours[i])
    # cv2.drawContours(img,contours,0,(0,255,0),2)
    # cv_show('img',img)
    
    
        car_contours = []
        for cnt in contours: 
            # 框选 生成最小外接矩形 返回值（中心(x,y), (宽,高), 旋转角度）rect[0]：矩形中心点坐标；rect[1]：矩形的高和宽；rect[2]：矩形的旋转角度
            rect = cv2.minAreaRect(cnt) 
            area_width, area_height = rect[1]
            #计算最小矩形的面积，初步筛选
            area = rect[1][0] * rect[1][1] #最小矩形的面积
        
            if area > MIN_AREA:
            # 选择宽大于高的区域
                if area_width < area_height:
                    area_width, area_height = area_height, area_width
                wh_ratio = area_width / area_height
            
            
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
                box=[]
                if wh_ratio > 2 and wh_ratio < 5.5:
                    found = True
                    if(debugMode):
                        print(rect)  
                        print('宽高:',rect[1])
                        print('面积：',area)
                        print('宽高比：',wh_ratio)
                    car_contours.append(rect)  # rect是minAreaRect的返回值，根据minAreaRect的返回值计算矩形的四个点
                    box = cv2.boxPoints(rect)  # box里面放的是最小矩形的四个顶点坐标
                    box = np.intp(box)  # 取整
                    if(debugMode):
                        print(box)
                
                    vertices = getVerticesFromBox(box)
                    img_return = np.copy(img)
                    oldimg = cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
                    cv_show('oldimg', oldimg)
                    # print(car_contours)
    return box, vertices, rect, img_return
                  
def seperate(ROI):
    
    seg1 = ROI[:,0:170]
    seg2 = ROI[:,170:340]
    seg3 = ROI[:,420:600]    
    seg4 = ROI[:,600:780]
    seg5 = ROI[:,780:960]
    seg6 = ROI[:,960:1140]
    seg7 = ROI[:,1140:1320]
    return seg1,seg2,seg3,seg4,seg5,seg6,seg7             
    
def getVerticesFromBox(box):
    for i in range(len(box)):
        # print('最小矩形的四个点坐标：', box[i])
        # 获取四个顶点坐标
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])
        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]
        vertices = np.array([[top_point_x, top_point_y], [bottom_point_x, bottom_point_y], [left_point_x, left_point_y],[right_point_x, right_point_y]])
        return vertices
 
 
def preprocess(debugMode = False):
    i = 1
    for item in os.listdir('./preprocess'): 
        imgpath = os.path.join('./preprocess',item)
        dst_dir = os.path.join('./result',str(i))
        i += 1
        if(~os.path.isdir(dst_dir)):
            os.mkdir(dst_dir) 
            
        if(debugMode):
            print("processing ",imgpath)
        box,vertices,rect,img = locate(imgpath,debugMode=False)
        if(debugMode):
            print(type(rect))
            print(rect)
        if (abs(rect[2])<10):
            angle = 0
        else:
            angle = -90
        rect_no_rotation = (rect[0],rect[1],angle)
        if(debugMode):
            print(rect_no_rotation)
        box_no_rotation = cv2.boxPoints(rect_no_rotation)
        box_no_rotation = np.int64(box_no_rotation) 
        if(debugMode):
            print(box_no_rotation) 
            print()
            print(box)
        src_vertices = vertices
        dst_vertices = getVerticesFromBox(box_no_rotation)
        src_pts = np.array([src_vertices[0],src_vertices[1],src_vertices[2]],dtype=np.float32)
        dst_pts = np.array([dst_vertices[0],dst_vertices[1],dst_vertices[2]],dtype=np.float32) 
        # 获取仿射变换矩阵  
        M = cv2.getAffineTransform(src_pts, dst_pts)
        print(M)
        dst = cv2.warpAffine(img, M, (img.shape[1],img.shape[0]))
        cv2.normalize(dst,dst,0,255,norm_type=cv2.NORM_MINMAX)
        if(debugMode):
            cv_show('img',img)
            cv_show('dst',dst)
        print(dst)
        dst_marked = cv2.drawContours(img ,[box_no_rotation],0,(0,255,0),2)
        if(debugMode):
            cv_show('dst_marked',dst_marked)
        miny = np.min(box_no_rotation[:,0])
        maxy = np.max(box_no_rotation[:,0])
        minx = np.min(box_no_rotation[:,1])
        maxx = np.max(box_no_rotation[:,1])
        if(debugMode):
            print(miny,maxy,minx,maxx)
        ROI = dst[minx:maxx,miny:maxy]
        ROI = cv2.resize(ROI,(440,140))
        ROI = cv2.resize(ROI,dsize=None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
        seg1,seg2,seg3,seg4,seg5,seg6,seg7=seperate(ROI)
        if(debugMode):
            cv_show('ROI',ROI)
            cv_show('seg',seg1)
            cv_show('seg',seg2)
            cv_show('seg',seg3)
            cv_show('seg',seg4)
            cv_show('seg',seg5)
            cv_show('seg',seg6)
            cv_show('seg',seg7)
        cv2.imwrite(dst_dir+"/seg1.jpg",seg1)
        cv2.imwrite(dst_dir+"/seg2.jpg",seg2)
        cv2.imwrite(dst_dir+"/seg3.jpg",seg3)
        cv2.imwrite(dst_dir+"/seg4.jpg",seg4)
        cv2.imwrite(dst_dir+"/seg5.jpg",seg5)
        cv2.imwrite(dst_dir+"/seg6.jpg",seg6)
        cv2.imwrite(dst_dir+"/seg7.jpg",seg7)
    
if __name__ == '__main__':
    preprocess(debugMode=True)
        
        
        
        
         
        
           