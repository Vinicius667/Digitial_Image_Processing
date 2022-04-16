import sys,os
from cv2_scripts import imshow
from numpy import float64,array,uint8,ones
from utils_scripts import zoom,butterworth2d,fspecial
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from numpy import asarray, double,cumsum,sum
from matplotlib import pyplot as plt
import cv2

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")


img_name = "brain"  #image name 
img_format = ".jpg" # image format
img_filename = img_name + img_format #[[[]]]]file name 
img = cv2.imread(img_filename ,0) # image uint8 vector 
imshow(img)

F = fftshift(fft2(asarray(img,dtype = double)))
H = butterworth2d(F.shape,40,type="lowpass")
Ff = F*H
img_f = asarray(abs(ifft2(ifftshift(F*H))),uint8)
imshow(img_f,"Imagem_bw",save=True)

img_f2 = cv2.medianBlur(img_f,15)
imshow(img_f2,"Imagem_median",save=True)

hist = cv2.calcHist([img_f2], [0], None, [255], [0,255])
plt.plot(hist)
plt.grid()
plt.xlim(0,255)
plt.title("Histrograma da imagem")
plt.xlabel("Pixel")
plt.ylabel("Quantidade")
plt.savefig("hist.png")


img_bin = cv2.threshold(img_f2, 75, 255, cv2.THRESH_BINARY)[1]
imshow(img_bin,"Imagem_bin",save=True)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
img_opened = cv2.morphologyEx(img_bin,cv2.MORPH_OPEN, kernel)
imshow(img_opened,"Imagem_opened",save=True)

contorno,hierarquia = cv2.findContours(img_opened,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
image_copy = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
cv2.drawContours(image=image_copy, contours=contorno, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
imshow(image_copy,"Imagem_contorno",save=True)

