import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import locating


provincelist = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"]

wordlist = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"]

# --- 绘制边界框
# --- 绘制矩形框
def DrawBox(im, box):
    cv2.rectangle(im, tuple(box[0]), tuple(box[1]), (255, 255, 255), 3)

# --- 绘制关键点
def DrawPoint(im, points):
    for p in points:
        cv2.circle(im, (p[0], p[1]), 5, (0, 0, 255), -1)

# --- 绘制车牌
def DrawLabel(im, label):
    im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(im)
    draw.text((30,30), label, font=ImageFont.truetype('yahei_mono_0.ttf', 40), fill=(255,0,0))
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return im

# --- 图片可视化
def ImgShow(imgpath, box, points, label):
    im = cv2.imread(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    im = DrawLabel(im, label)
    cv2.imshow('img', im)
    
    
def get_dst_points(box):
    top_left = box[0]
    bottom_right = box[1]
    top_right = [bottom_right[0], top_left[1]]
    bottom_left=  [top_left[0],bottom_right[1]]
    dst = np.array([top_left,top_right,bottom_left,bottom_right],dtype=np.float32)
    return dst

# 投影变换
def rectification(imgpath, src_points, dst_points):
    src_points = np.array([src_points[0],src_points[1],src_points[3],src_points[2]], dtype=np.float32)
    print(src_points,"\n")
    print(dst_points,"\n")
    img = cv2.imread(imgpath)
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
    return corrected_img

def get_license_info(imgpath):
    # 图像路径（根据数据库的实际存放路径更改）

    # 图像名
    imgname = os.path.basename(imgpath).split('.')[0]

    # 根据图像名分割标注
    _, _, box, points, label, brightness, blurriness = imgname.split('-')

    # --- 边界框信息
    box = box.split('_')
    box = [list(map(int, i.split('&'))) for i in box]

    # --- 关键点信息
    points = points.split('_')
    points = [list(map(int, i.split('&'))) for i in points]
    # 将关键点的顺序变为从左上顺时针开始
    points = points[-2:]+points[:2]

    # --- 读取车牌号
    label = label.split('_')

    # 省份缩写
    province = provincelist[int(label[0])]

    # 车牌信息
    words = [wordlist[int(i)] for i in label[1:]]

    # 车牌号
    label = province+''.join(words)

    information = {'imgname': imgname, 'box': box, 'points': points, 'label':label}
    return  information

def get_ROI(imgpath, box):
    img = cv2.imread(imgpath)
    print(img)
    top_left, bottom_right = box[0], box[1]
    left, top = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
    right, bottom = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
    print(img is None)
    # 提取矩形区域
    cropped_image = img[top:bottom, left:right]

    return cropped_image

def seperate(ROI):
    seg1 = ROI[:,0:180]
    seg2 = ROI[:,180:360]
    seg3 = ROI[:,400:580]    
    seg4 = ROI[:,580:750]
    seg5 = ROI[:,750:920]
    seg6 = ROI[:,920:1090]
    seg7 = ROI[:,1090:1320]
    return seg1,seg2,seg3,seg4,seg5,seg6,seg7


def preprocess(img_directory):
    i = 1
    for item in os.listdir(img_directory): # 遍历访问待处理图片路径下的每张图片
        item_path = os.path.join(img_directory,item)
        result_dir = './result'
        folder_path = os.path.join(result_dir,str(i))
        i+=1
        # if(~os.path.isdir(folder_path)):
        #     os.mkdir(folder_path) # 单张图片的预处理结果都放在这个文件夹里（包含7个.jpg文件，7个字符）
        img = cv2.imread(item_path)
        img = cv2.resize(img, (400, int(400 * img.shape[0] / img.shape[1])))
        img_copy = img.copy()
        rect = locating.find_plates(img)
        plate = img_copy[rect[1]-5:rect[3]+5,rect[0]-5:rect[2]+5]
        cv2.imshow('plate',plate)
        # cv2.imshow('img',img)
        # cv2.rectangle(img, (rect[0] - 5, rect[1] - 5), (rect[2] + 5, rect[3] + 5), (0, 255, 0), 2)
        # cv2.imshow('after', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def ccpd():
    name = '01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg'
    imgpath = '/mnt/d/CCPD2019/ccpd_base/'+name
    information = get_license_info(imgpath)
    [imgname, box, points, label] = [information.get('imgname'), information.get('box'), information.get('points'), information.get('label')]
    print(points)
    print(label)
    print(box)
    print(imgname)

    ROI = get_ROI(imgpath,box)
    img_rect = rectification(imgpath,points,get_dst_points(box))
    # 显示
    ImgShow(imgpath, box, points, label)
    tmp_filename = "/home/zhang/DIP/Sustech-2024-DIP-project/preprocess/" + imgname + ".jpg"
    print(tmp_filename)
    cv2.imwrite(tmp_filename, img_rect)
    ROI_corrected = get_ROI(tmp_filename,box)
    ROI_gray = cv2.resize(ROI_corrected,(440,140))
    ROI_gray=  cv2.resize(ROI_gray,dsize=None, fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    ROI_gray = cv2.cvtColor(ROI_gray, cv2.COLOR_BGR2GRAY)
    cv2.imshow('ROI', ROI_gray)
    #ROI_gray = cv2.Sobel(ROI_gray, cv2.CV_64F, 1, 0, ksize=5)
    _,ROI_gray = cv2.threshold(ROI_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ROI_gray = cv2.normalize(ROI_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    word1,word2,word3,word4,word5,word6,word7 = seperate(ROI_gray)
    #ROI_gray = cv2.equalizeHist(ROI_gray)
    #ROI_gray = cv2.resize(ROI_gray, None, fx=4, fy=4, interpolation=None)

    #ROI_gray = cv2.Laplacian(ROI_gray, cv2.CV_64F, dst=None, ksize=5)
    #ROI_gray = cv2.normalize(ROI_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imshow('ROI_threshold',ROI_gray)
    cv2.imshow('img_rect',img_rect)
    cv2.imshow('seg1',word1)
    cv2.imshow('seg2',word2)
    cv2.imshow('seg3',word3)
    cv2.imshow('seg4',word4)
    cv2.imshow('seg5',word5)
    cv2.imshow('seg6',word6)
    cv2.imshow('seg7',word7)
    cv2.waitKey()


preprocess('/home/zhang/Sustech-2024-DIP-project/preprocess')

