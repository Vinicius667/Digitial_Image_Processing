import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

from cv2_scripts import imshow
import numpy as np
import cv2
from numpy.random import randint


img_name = "onion"  #img name 
img_format = ".png" # img format
img_filename = img_name + img_format #file name 
img = cv2.imread(img_filename ,1) # img uint8 vector
pixel_values = img.reshape((-1, 3))
imshow(img)
pixel_values = np.float32(pixel_values) 
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1) # creterias for k-means


for k in range(3,11,2):
    ret, labels, centers = cv2.kmeans(pixel_values, k,None, criteria, 30, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers) # centroids
    segmented_img = centers[labels.flatten()] # segmented image
    segmented_img = segmented_img.reshape(img.shape) # reshape vector
    imshow(segmented_img,f"onion_k_{k}",save=True) 



# Legumes : (k, [cor1,cor2,...],corrigir_seg) 
pimenta = (9,([112,37],),False) 
cebola = (9,([15,128],[15,129]),False)
fruta_verde = (9,([50,113],[50,114],[50,90]),True)
fruta_amarela = (9,([134,197],[100,197]),True)
fruta_laranja = (9,([97,105],),False)
legumes = [pimenta,cebola,fruta_verde,fruta_amarela,fruta_laranja]

image_copy = img.copy()
for legume in legumes:
    k = legume[0] # Value o k
    corrigir_seg = legume[2] # Bool to check if needs correction
    segmented_img = cv2.imread(f"onion_k_{k}.png") 
    
    mask = np.zeros(segmented_img.shape[:2])
    for cor in legume[1]:
        cor = segmented_img[cor[0],cor[1],:]
        lower = upper = cor
        mask+=   cv2.inRange(segmented_img, lower, upper)
        imshow(mask)
    mask = mask.astype(dtype=np.uint8)
    if corrigir_seg:  # Apply erode and dialate to separate contours 
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE, kernel)
        mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE, kernel)
        
    contorno = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0] #find contour
    areas = [cv2.contourArea(c) for c in contorno] # areas of contour
    max_index = np.argmax(areas) # index of largest area
    cnt_larger=contorno[max_index] # contour with largest area
    cv2.drawContours(image=image_copy, contours=cnt_larger, contourIdx=-1, color = [randint(0,255),randint(0,255),randint(0,255)], thickness=2,lineType=cv2.LINE_AA)
imshow(image_copy,"Imagem_contorno",save=True)
