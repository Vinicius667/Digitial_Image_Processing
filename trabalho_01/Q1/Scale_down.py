# -*- coding: utf-8 -*-
from numpy import asarray,uint8,delete,linspace,uint16



def scale_down(img,f):
    
    rows,cols = img.shape # Number of rows and columns of the original image 
    
    rows_resized = int(f*rows)  # Number of rows of the resized image
    cols_resized = int(f*cols)  # Number of columns of the resized image
    
    """ Creates two arrays of evenly spaced numbers representing the rows and columns
    that will be removed. """
    delete_rows = linspace(0, rows, rows - rows_resized ,endpoint=False,dtype = uint16)
    delete_cols = linspace(0, cols, cols - cols_resized,endpoint=False,dtype = uint16)
    
    
    """ Remove the rows and columns of the original image according to the vectors
    delete_rows and delete_cols. """
    im_resized = delete(img,delete_rows,0)
    im_resized = delete(im_resized,delete_cols,1)
    
        
    return asarray(im_resized,dtype=uint8())