import numpy as np
import cv2 as cv

def preprocess(img):
    #灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #高斯模糊
    blur = cv.GaussianBlur(gray,(5,5),0)
    #Sobel算子边缘检测
    sobel = cv.Sobel(blur,cv.CV_8U,1,0,ksize=3)
    #二值化
    ret, binary = cv.threshold(sobel,0,255,cv.THRESH_OTSU + cv.THRESH_BINARY)
    #腐蚀膨胀
    element1 = cv.getStructuringElement(cv.MORPH_RECT,(17,5))
    element2 =cv.getStructuringElement(cv.MORPH_RECT,(20,1))
    dilation = cv.dilate(binary,element2,iterations=1)
    erosion = cv.erode(dilation,element1,iterations=1)
    dilation2 = cv.dilate(erosion,element2,iterations=3)
    #查找轮廓
    contours, hierachy = cv.findContours(dilation2, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #筛选出符合条件的轮廓
    rects = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv.contourArea(cnt)
        x, y,w,h = cv.boundingRect(cnt)
        if(w < 5 or h < 5 or w/h > 5 or h / w > 5 or area < 100):
            continue
        rect = cv.minAreaRect(cnt)
        rects.append(rect)
    return contours,rects

def get_ROI(img, rect):
    center = rect.center
    size = rect.size
    angle = rect.angle

    # 将角度转换为弧度
    theta = angle * (3.141592653589793238462643383279502884 / 180.0)

    # 创建旋转矩阵
    M = cv.getRotationMatrix2D(center, angle, 1.0)

    # 提取 ROI（感兴趣区域）
    # 注意：提取的 ROI 尺寸应该大到足够容纳整个旋转矩形
    roi_width = int(max(size))
    roi_height = int(min(size))
    roi = cv.getRectSubPix(img, (roi_width, roi_height), center)

    # 应用仿射变换以纠正旋转
    corrected_roi = cv.warpAffine(roi, M, (roi_width, roi_height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return corrected_roi


def find_license_plate(img,rects):
    # 提取旋转矩形的中心点、尺寸和角度信息
    for i in range(len(rects)):
        ROI = get_ROI(img, rects[i])
        median = np.median(ROI)



if __name__ == '__main__':
    img = cv.imread('../image_test.jpg')
    contours,rects = preprocess(img)

    # for i in range(len(rects)):
    #     box = cv.boxPoints(rects[i])
    #     box = np.int0(box)
    #     cv.drawContours(img, [box], 0, (0, 255, 0), 2)
    box = cv.boxPoints(rects[8])
    box = np.int0(box)
    cv.drawContours(img, [box], 0, (0, 255, 0), 2)
    # cv.drawContours(img, contours, -1, (0,255,0), thickness=None, lineType=None, hierarchy=None, maxLevel=None,offset=None)
    cv.imshow('img',img)
    cv.imwrite('../image_test_processed.jpg',img)
    print(len(rects),len(rects[0]),len(rects[0][0]))
    cnt = 0
    for rect in rects:
        print(cnt," ",rect)
        print("\n")
        cnt +=1
    cv.waitKey(0)