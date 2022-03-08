
import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

from cv2 import imread,imwrite
from cv2_scripts import imshow
from Scale_down import scale_down
from Scale_up import scale_up
from numpy import float64,array

"""
 This script downscales an image using nearest neighbor and bilinear algorithm
"""

img_filename = "notre-dame"  #image filename 
img = imread(img_filename + ".pgm",0) # image uint8 vector 
imwrite(f"{img_filename}.png" , img) # convertes orginal image to png 



fs = array([0.1,10],dtype = float64) # scale factor



for f in fs:
    if f < 1:
        img = scale_down(img, f)
    elif f > 1:
        img = scale_up(img, f)
    else:
        img = img
    imshow(img,f"{img_filename}_teste_{str(f)}.png",save=False)