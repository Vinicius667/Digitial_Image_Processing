# -*- coding: utf-8 -*-
import cv2
from numpy import asarray, log10, max, min, uint8


def imshow(mat,winname="Imagem",normalize = False,log=False,c=110,save=False,show=True):
    mat =  asarray(mat+0.5,dtype=uint8)
    if log:
        mat = c*log10(1+mat)
    if normalize:
        mat_non_neg = mat - min(mat)
        mat = 255*(mat_non_neg)/max(mat_non_neg)
    if show:
        cv2.imshow(winname,mat)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    
    if save:
        cv2.imwrite(winname+".png", mat)

