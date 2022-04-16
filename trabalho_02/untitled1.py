# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 12:11:21 2022

@author: almei
"""
import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")
from cv2 import imread,imwrite,MORPH_OPEN,MORPH_CLOSE,morphologyEx,filter2D,medianBlur
from cv2_scripts import imshow

img = imread('opencv_logo.jpg',0)
final = medianBlur(img, 3)
imshow(final)