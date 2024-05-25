import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

name = '01-86_91-298&341_449&414-458&394_308&410_304&357_454&341-0_0_14_28_24_26_29-124-24.jpg'
imgpath = '/mnt/d/CCPD2019/ccpd_base/'+name
img = cv2.imread(imgpath)
cv2.imshow('original',img)
cv2.waitKey(0)