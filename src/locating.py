# 导库
import cv2 as cv
import numpy as np
import os


# 方法
# 导入图片资源 path为路径
def load_image(path):
    src = cv.imread(path)
    return src


# 灰度拉伸方法
'''
它可以有选择的拉伸某段灰度区间以改善输出图像,如果一幅图像的灰度集中
在较暗的区域而导致图像偏暗，可以用灰度拉伸功能来拉伸(斜率>1)物体灰度区间以改善图像；同样如果图像灰度集中在较亮的区域而导致图像偏亮，也可以用灰
度拉伸功能来压缩(斜率<1)物体灰度区间以改善图像质量。
灰度拉伸
定义：灰度拉伸，也称对比度拉伸，是一种简单的线性点运算。作用：扩展图像的
      直方图，使其充满整个灰度等级范围内
公式：
g(x,y) = 255 / (B - A) * [f(x,y) - A],
其中，A = min[f(x,y)],最小灰度级；B = max[f(x,y)],最大灰度级；
     f(x,y)为输入图像,g(x,y)为输出图像
缺点：如果灰度图像中最小值A=0，最大值B=255，则图像没有什么改变
'''


def gray_stretch(image):
    max_value = float(image.max())
    min_value = float(image.min())
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = (255 / (max_value - min_value) * image[i, j] - (255 * min_value) / (max_value - min_value))
    return image


'''
图像整体二值化
'''


def image_binary(image):
    max_value = float(image.max())
    min_value = float(image.min())
    '''
    这里利用自适应二值化以及自动求阈值的方法明显效果不好。因此设置阈值这样设置的效果很容易想到，当图片为一张纯色图时阈值为对应像素值，当图包含
    255与0时，阈值为122，总体的适应的效果会比较好。方法返回二值图
    '''
    ret = max_value - (max_value - min_value) / 2
    ret, thresh = cv.threshold(image, ret, 255, cv.THRESH_BINARY)
    return thresh


'''
矩形轮廓角点，寻找到矩形之后记录角点，用来做参考以及画图。
'''


def find_rectangle(contour):
    y, x = [], []
    for value in contour:
        y.append(value[0][0])
        x.append(value[0][1])
    return [min(y), min(x), max(y), max(x)]


'''
车牌定位方法，需要两个参数，第一个是用来寻找位置，第二个为原图，用来绘制矩形。寻找位置的图片为经过几次形态学操作的图片。这里利用权值的操作，实
现了定位的最高概率。
'''


def loacte_plate(image, after):
    '''
    定位车牌号
    '''
    # 寻找轮廓
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    img_copy = after.copy()
    # 找出最大的三个区域
    solving = []
    for c in contours:
        r = find_rectangle(c)
        '''
        这里就算出面积和长宽比
        '''
        a = (r[2] - r[0]) * (r[3] - r[1])
        s = (r[2] - r[0]) / (r[3] - r[1])

        solving.append([r, a, s])
    # 通过参考选出面积最大的区域
    solving = sorted(solving, key=lambda b: b[1])[-3:]
    # 颜色识别
    maxweight, maxindex = 0, -1
    for i in range(len(solving)):  #
        wait_solve = after[solving[i][0][1]:solving[i][0][3], solving[i][0][0]:solving[i][0][2]]
        # BGR转HSV
        hsv = cv.cvtColor(wait_solve, cv.COLOR_BGR2HSV)
        # 蓝色车牌的范围 Hsv色彩空间的设置。
        lower = np.array([100, 50, 50])
        upper = np.array([140, 255, 255])
        # 利用inrange找出掩膜
        mask = cv.inRange(hsv, lower, upper)
        # 计算权值用来判断。
        w1 = 0
        for m in mask:
            w1 += m / 255
        w2 = 0
        for n in w1:
            w2 += n
        # 选出最大权值的区域
        if w2 > maxweight:
            maxindex = i
            maxweight = w2
    return solving[maxindex][0]


'''
框出车牌 获取位置坐标，并返回图像
'''


# 对图像的预处理
def find_plates(image):
    # image = cv.resize(image, (400, int(400 * image.shape[0] / image.shape[1])))
    # 转换为灰度图像
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # 灰度拉伸
    # 如果一幅图像的灰度集中在较暗的区域而导致图像偏暗，可以用灰度拉伸功能来拉伸(斜率>1)物体灰度区间以改善图像；
    # 同样如果图像灰度集中在较亮的区域而导致图像偏亮，也可以用灰度拉伸功能来压缩(斜率<1)物体灰度区间以改善图像质量
    stretchedimage = gray_stretch(gray_image)  # 进行灰度拉伸，是因为可以改善图像的质量

    '''进行开运算，用来去除噪声'''
    # 构造卷积核

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (30, 30))
    # 开运算
    openingimage = cv.morphologyEx(stretchedimage, cv.MORPH_OPEN, kernel)
    # 获取差分图，两幅图像做差  cv2.absdiff('图像1','图像2')
    strtimage = cv.absdiff(stretchedimage, openingimage)

    # 图像二值化
    binaryimage = image_binary(strtimage)
    # canny边缘检测
    canny = cv.Canny(binaryimage, binaryimage.shape[0], binaryimage.shape[1])
    # 5 24效果最好
    kernel = np.ones((5, 24), np.uint8)
    closingimage = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    openingimage = cv.morphologyEx(closingimage, cv.MORPH_OPEN, kernel)
    # 11 6的效果最好
    kernel = np.ones((11, 6), np.uint8)
    openingimage = cv.morphologyEx(openingimage, cv.MORPH_OPEN, kernel)
    # 消除小区域，定位车牌位置
    rect = loacte_plate(openingimage, image)  # rect包括轮廓的左上点和右下点，长宽比以及面积
    # 展示图像
    cv.imshow('image', image)
    cv.rectangle(image, (rect[0] - 5, rect[1] - 5), (rect[2] + 5, rect[3] + 5), (0, 255, 0), 2)
    cv.imshow('after', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return rect

def running():
    file_path = './preprocess'
    for filewalks in os.walk(file_path):
        for files in filewalks[2]:
            print('正在处理', os.path.join(filewalks[0], files))
            find_plates(load_image(os.path.join(filewalks[0], files)))


if __name__ == '__main__':
    running()
