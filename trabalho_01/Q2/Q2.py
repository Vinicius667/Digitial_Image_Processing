import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

from cv2 import imread
from cv2_scripts import imshow
from numpy import asarray, double
from numpy.fft import fft2,fftshift,ifft2,ifftshift
from utils_scripts import hmf,plot3d


yl = 0.9
yh = 1.4
c = 0.5
D0 = 50


img = imread("image2" + ".jpg",0)
F = fftshift(fft2(asarray(img,dtype = double)))
imshow(abs(F),'FFT',normalize=True,log=True,save=True)
H = hmf(F.shape,yh=yh,yl=yl,c=c,D0=D0)
plot3d(H,'Filtro_homomorfico3d',"Filtro homomórfico",save=True)
imshow(H,'Filtro_homomorfico',normalize=True,log=False,save=True)
Ff = F*H
imshow(abs(Ff),'FFT_filtrada',normalize=True,log=True,save=True)

img_f = abs(ifft2(ifftshift(F*H)))

imshow(img_f,f'Imagem_filtrada_yh_{yh}_yl_{yl}_c_{c}_D0_{D0}',normalize=False,log=False,save=True)