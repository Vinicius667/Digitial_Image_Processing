from numpy import array, ogrid, ones, sqrt,e
import matplotlib.pyplot as plt
from matplotlib import cm
from cv2 import imread

plt.ioff()

def zoom(img,z):
    h,w =  img.shape
    cy = int(h/2)
    cx = int(w/2)
    dy = int(h/(2*z))
    dx = int(w/(2*z))
    print(cx,dx,cy,dy)
    return img[cy-dy:cy+dy,cx-dx:cx+dx]

def plot3d(mat,filename="imagem",legend="Plot 3d",save=False,show=False): #Plots images in 3d
    h,w = mat.shape
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(4, 4), dpi=150)
    Y, X = ogrid[:h, :w]
    #ax.zaxis.set_major_locator(LinearLocator(5))
    #ax.xaxis.set_major_locator(LinearLocator(5))
    ax.zaxis.set_major_formatter('{x:.02f}')
    surf = ax.plot_surface(X, Y, mat, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False,label=legend)
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d
    
    if legend:
        ax.legend()
    
    if show:
        plt.show()
    
    if save:
        plt.savefig(filename+".png")
    

def fspecial(type:str,*args): # Still in progress 
    if type == 'average':
        if args:
            fsize = args[0]
        else:
            fsize = 3
        f = ones((fsize,fsize))/fsize**2

    
    if type == "laplacian":
        if args:
            alpha = args[0]
        else:
            alpha = 0.2
    
        f = array([   [alpha,1-alpha,alpha],
                      [1-alpha,-4,1-alpha]
                   ,  [alpha,1-alpha,alpha]])/(alpha+1)
    return f 


def butterworth2d(mat_shape,D0=None,center = None,n=2,W=None,type = "lowpass"): #Implements Butterworth filter
    h,w = mat_shape    
    if not center:
        center = (int(h/2), int(w/2))
    Y, X = ogrid[:h, :w]
    if type not in ["notch_highpass","notch_lowpass"]:
        D = sqrt( (Y - center[0])**2 + (X - center[1])**2) 
   
    if type == "lowpass":
        H = 1/(1 +  ( D/D0 )**2*n )
    if type == "highpass":
        H = 1/(1 +  ( D0/D )**2*n )
    if type == "bandstop":
        H = 1/(1 +  (W*D/(D**2 -  D0**2) )**2*n )
    if type == "bandpass":
        H =  1 - (1/(1 +  (W*D/(D**2 -  D0**2) )**2*n ))    
    if type == "notch_lowpass":
        D = sqrt( (Y - h/2 - center[0])**2 + (X - w/2 - center[1])**2) 
        H = 1/(1 +  ( D/D0 )**2*n )
    if type == "notch_highpass":
        D = sqrt( (Y - h/2 - center[0])**2 + (X - w/2 - center[1])**2) 
        H = 1/(1 +  ( D0/D )**2*n )
    return H
       
def gaussian2d(mat_shape,D0=None,center = None,W=None,type = "lowpass"): #Implements gaussian filter
    h,w = mat_shape    
    if not center:
        center = (int(h/2), int(w/2))
    Y, X = ogrid[:h, :w]
    D = sqrt( (Y - center[0])**2 + (X - center[1])**2) 
    if type == "lowpass":
        H = e**-((D**2)/(2*(D0**2)))
    if type == "highpass":
        H = 1 - e**-((D**2)/(2*(D0**2)))
    if type == "bandstop":
        H = 1 - e**-(((D**2) - (D0**2))/(D*W))**2
    if type == "bandpass":
        H =   e**-(((D**2) - (D0**2))/(D*W))**2
    
    return H

def hmf(mat_shape,yh,yl,c,D0,center=None): #Implements homomorphic filter
    h,w = mat_shape    
    if not center:
        center = (int(h/2), int(w/2))
    Y, X = ogrid[:h, :w]
    D = sqrt( (Y - center[0])**2 + (X - center[1])**2)
    H = (yh-yl)*(1-(e**(-(c*((D**2)/(D0**2)))))) + yl
    return H


if __name__ == "__main__":
    img = imread("./Q1/notre-dame.pgm",0)
    H = butterworth2d(img.shape,80,type="lowpass")
    plot3d(H,show=True,legend="Filtro Butterworth")