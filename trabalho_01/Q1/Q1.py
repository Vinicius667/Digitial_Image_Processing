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
from utils_scripts import zoom
"""
 This script downscales an image and upscales using nearest neighbor and bilinear algorithm
 for different scale factors.
"""

img_filename = "notre-dame"  #image filename 
img = imread(img_filename + ".pgm",0) # image uint8 vector 
imwrite(f"{img_filename}.png" , img) # convertes orginal image to png 
zoom2x = zoom(img, 2) #crops imagem to zoom 2x 
zoom8x = zoom(img, 8) #crops imagem to zoom 8x
imwrite(f"zoom8x_1.0.png", zoom8x ) # saves zoomed image
imwrite(f"zoom2x_1.0.png", zoom2x ) # saves zoomed image

types = ["bilinear","nearest_neighbor"] # types array
fs = array([0.2,0.5,1.4,2],dtype = float64) # scale factor array



for t in types:  # Itarate over types array
    for f in fs: # Itarate over scale factor array
        if f < 1: 
            im_resized = scale_down(img, f)
            imshow(im_resized,f"{img_filename}_{str(f)}.png",save=True)
            zoom2x = zoom(im_resized, 2)
            imwrite(f"zoom2x_{str(f)}.png", zoom2x )
        elif f > 1:
            im_resized = scale_up(img, f,t)
            imshow(im_resized,f"{img_filename}_{str(f)}_{t}.png",save=True)
            h,w =  im_resized.shape
            zoom8x = zoom(im_resized, 8)
            imwrite(f"zoom8x_{str(f)}_{t}.png", zoom8x )
        else:
            im_resized = img