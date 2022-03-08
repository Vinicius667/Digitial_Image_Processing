from numpy import int16, asarray,linspace,ones,insert,copy,uint8,float64,ones
from Find_nearest import find_nearest
from numba import njit
from cv2 import imread
from cv2_scripts import imshow
from numpy import float64
import sys,os
abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
os.chdir(dir_name)
sys.path.append("../")

def scale_up(img,f:float64,type="bilinear"):    
    rows,cols = img.shape
    
    rows_resized = int(f*rows) # Number of rows of the resized image 
    cols_resized = int(f*cols) # Number of columns of the resized image 
    img  = asarray(img,dtype=int16)  # Converts array to int16 in order to be able to add (-1)
    
    """ Creates two arrays of evenly spaced numbers representing the rows and columns
    that will be inserted. """
    
    
    insert_rows = linspace(1, rows-1, abs(rows - rows_resized)  ,endpoint=False,dtype = int16)
    insert_cols = linspace(1, cols-1, abs(cols - cols_resized)  ,endpoint=False,dtype = int16)
        
    if type == "bilinear":

        row_insert = -1*ones(cols,dtype = int16) # row filled with -1 that will be replaced later
        im_dummy = insert(img,insert_rows,row_insert,0) # dummy array with lots of -1 in the inserrted rows
        col_insert = -1*ones((rows + len(insert_rows),len(insert_cols)),dtype = int16)
        im_dummy = insert(im_dummy,insert_cols,col_insert,1) # dummy array with lots of -1 in the inserrted columns
        N,M =  im_dummy.shape 
        im_resized = fill_bilinear(im_dummy,im_dummy,N,M,f)
    elif type == "nearest_neighbor":
        count_insert = 0
        im_dummy = ones((rows_resized,cols))
        for y in range(rows):
            im_dummy[y+count_insert,:] =  img[y,:]
            try:
                while True:
                    if y == insert_rows[count_insert]:
                        count_insert +=1
                        im_dummy[y+count_insert,:] =  img[y,:]
                    else:
                        break
            except:
                pass
        im_resized = ones((rows_resized,cols_resized))
        count_insert = 0   
        for x in range(cols):
            im_resized[:,x+count_insert] =  im_dummy[:,x]
            try:
                while True:
                    if x == insert_cols[count_insert]:
                        count_insert +=1
                        im_resized[:,x+count_insert] =  im_dummy[:,x]
                    else:
                        break
            except:
                pass

                
    return asarray(im_resized,dtype = uint8)


@njit
def fill_bilinear(im_resized,im_dummy,N,M,f):
    im_resized = copy(im_dummy)
    N,M =  im_resized.shape    
    for x in range(M):
        for y in range(N):
            if im_dummy[y,x] == -1:
                im_resized[y,x] = find_nearest(im_dummy,x,y,f)
    return im_resized
       

if __name__ == "__main__":
    img = imread("notre-dame.pgm",0)
    #img_resized = scale_up(img,2)
    #imshow(img_resized)
    img_resized = scale_up(img,2,"nearest_neighbor")
    imshow(img_resized)