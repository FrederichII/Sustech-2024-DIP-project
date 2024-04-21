from PIL import Image, ImageDraw, ImageFont
import os


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


def DrawBox(im, box):
    draw = ImageDraw.Draw(im)
    draw.rectangle([tuple(box[0]), tuple(box[1])],  outline="#FFFFFF", width=3)

# --- 绘制四个关键点


def DrawPoint(im, points):

    draw = ImageDraw.Draw(im)

    for p in points:
        center = (p[0], p[1])
        radius = 5
        right = (center[0]+radius, center[1]+radius)
        left = (center[0]-radius, center[1]-radius)
        draw.ellipse((left, right), fill="#FF0000")

# --- 绘制车牌


def DrawLabel(im, label):
    draw = ImageDraw.Draw(im)
   # draw.multiline_text((30,30), label.encode("utf-8"), fill="#FFFFFF")
    font = ImageFont.truetype('simsun.ttc', 64)
    draw.text((30, 30), label, font=font)

# --- 图片可视化


def ImgShow(imgpath, box, points, label):
    # 打开图片
    im = Image.open(imgpath)
    DrawBox(im, box)
    DrawPoint(im, points)
    DrawLabel(im, label)
    # 显示图片
    im.show()
    im.save('result.jpg')


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

    # --- 图片可视化
    information = {'imgname': imgname, 'box': box, 'points': points, 'label':label}
    return  information


if __name__ == '__main__':
    imgpath = 'D:/CCPD2019/ccpd_base/01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg'
    information = get_license_info(imgpath)
    [box, points, label] = [information.get('box'), information.get('points'), information.get('label')]
    ImgShow(imgpath, box, points, label)

