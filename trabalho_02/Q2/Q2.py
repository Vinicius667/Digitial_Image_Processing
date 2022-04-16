import sys,os
from cv2_scripts import imshow
from numpy import float64,array,uint8,ones
from utils_scripts import zoom,butterworth2d,fspecial
import numpy as np
from matplotlib import pyplot as plt
import cv2

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    cnt_scaled = cnt_scaled.astype(np.int32)

    return cnt_scaled


img_name = "onion"  #img name 
img_format = ".png" # img format
img_filename = img_name + img_format #[[[]]]]file name 
img = cv2.imread(img_filename ,1) # img uint8 vector
pixel_values = img.reshape((-1, 3))
imshow(img)
pixel_values = np.float32(pixel_values)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

'''
for k in range(3,9,2):
    ret, labels, centers = cv2.kmeans(pixel_values, k,None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    imshow(segmented_img,f"onion_k_{k}",save=True)
    np.random.seed(1)
'''


k = 5
segmented_img = cv2.imread(f"onion_k_{k}.png")
pimenta = (9,(segmented_img[112,37,:]))
'''
#pimenta 

lower = upper = pimenta[1]
mask = cv2.inRange(segmented_img, lower, upper)
imshow(mask,show=True)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
mask = cv2.morphologyEx(mask,cv2.MORPH_ERODE, kernel)
imshow(mask,show=True)
contorno = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
areas = [cv2.contourArea(c) for c in contorno]
max_index = np.argmax(areas)
cnt_larger=contorno[max_index]
cnt_larger = scale_contour(cnt_larger,1.2)
image_copy = img.copy()
cv2.drawContours(image=image_copy, contours=cnt_larger, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
imshow(image_copy,"Imagem_contorno",show=True)
'''
cebola = (9,(segmented_img[15,128,:],segmented_img[15,129,:]))

legumes = [pimenta,cebola]
for legume in legumes:
    k = legume[0]
    segmented_img = cv2.imread(f"onion_k_{k}.png")
    
    mask = np.zeros(segmented_img.shape[:2])
    for cor in legume[1]:
        lower = upper = cor
        mask+=   cv2.inRange(segmented_img, lower, upper)
    mask = mask.astype(dtype=np.uint8)
    imshow(mask,show=True)
    contorno = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    areas = [cv2.contourArea(c) for c in contorno]
    max_index = np.argmax(areas)
    cnt_larger=contorno[max_index]
    
    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=cnt_larger, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
    imshow(image_copy,"Imagem_contorno",show=True)





"""

for i in range(k):
    print(i)
    cluster_img = np.zeros(img[:,:,1].shape).flatten()
    cluster_img[labels.flatten() == i] = 255
    cluster_img = cluster_img.reshape(img[:,:,1].shape)
    imshow(cluster_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cluster_img = cv2.morphologyEx(cluster_img,cv2.MORPH_CLOSE, kernel)
    imshow(cluster_img.reshape(img[:,:,1].shape))
    cluster_img = cv2.morphologyEx(cluster_img,cv2.MORPH_OPEN, kernel)
    imshow(cluster_img)
    
"""