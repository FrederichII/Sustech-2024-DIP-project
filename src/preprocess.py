import os
import cv2
import freetype
import numpy as np
from PIL import Image, ImageDraw, ImageFont



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
    draw.text((30,30), label, font=ImageFont.truetype('msyh.ttc', 40), fill=(255,0,0))
    im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return im

# --- 图片可视化
def ImgShow(imgpath, box, points, label):
    im = cv2.imread(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    im = DrawLabel(im, label)
    cv2.imshow('img', im)



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
    top_left, bottom_right = box[0], box[1]
    left, top = min(top_left[0], bottom_right[0]), min(top_left[1], bottom_right[1])
    right, bottom = max(top_left[0], bottom_right[0]), max(top_left[1], bottom_right[1])
    # 提取矩形区域
    cropped_image = img[top:bottom, left:right]

    return cropped_image

if __name__ == '__main__':
    imgpath = 'D:/CCPD2019/ccpd_base/01-90_87-240&501_441&563-437&567_237&561_230&490_430&496-0_0_2_12_32_26_26-180-20.jpg'
    information = get_license_info(imgpath)
    [imgname, box, points, label] = [information.get('imgname'), information.get('box'), information.get('points'), information.get('label')]
    print(points)
    print(label)
    print(box)
    print(imgname)
    ROI = get_ROI(imgpath,box)

    # 显示
    ImgShow(imgpath, box, points, label)
    cv2.imshow('ROI',ROI)
    cv2.waitKey()


