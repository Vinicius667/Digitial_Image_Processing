import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

from cv2_scripts import imshow
from cv2 import imread
from numpy import asarray, double, ones
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from utils_scripts import butterworth2d,plot3d

img_filename = "moire"
img = imread(img_filename+ ".tif",0)



imshow(img,"Imagem original",save=True)


F = fftshift(fft2(asarray(img,dtype = double))) # FFT of original image
imshow(abs(F),'FFT',normalize=True,log=True,save=True)


H = ones(F.shape)

notches = [(10,39,30),   # Array of notches
           (10,-39,30),
           (5,78,30),
           (5,78,30),]


"""
    Implements homomorphic filter using Butterworh filter
"""
for d0,uk,vk in notches:
    H *= butterworth2d(F.shape,D0=d0,center=(uk,vk),type="notch_highpass") 
    H *= butterworth2d(F.shape,D0=d0,center=(uk,-vk),type="notch_highpass")

imshow(H,'Filtro_rejeita_notch',normalize=True,log=False,save=True)
plot3d(H,'Filtro_rejeita_notch','Filtro rejeita notch',save=True)


Ff = F*H 
imshow(abs(Ff),'FFT_filtrada',normalize=True,log=True,save=True)


img_f = abs(ifft2(ifftshift(F*H)))


imshow(img_f,'Imagem_filtrada',normalize=True,log=False,save=True)




