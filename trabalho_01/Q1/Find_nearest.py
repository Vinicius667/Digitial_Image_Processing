# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:28:08 2022

@author: almei
"""

from numpy import array,ceil
from numba import jit



@jit(nopython=True)
def    find_nearest(img,x,y,f):
    p = int(ceil(f))
    N,M =  img.shape
    soma = 0
    soma_dist = 0
    
    if x!= M-1 and y!=N-1:
        #superior esquerdo
        near_found = False
        for i in range(1,p+1):
            idx = x-i
            if idx < 0:
                idx = 0
            for j in range(1,p+1):
                idy = y-j
                if idy < 0:
                    idy = 0
                if img[idy,idx] > -1:
                    near_found = True
                    #print(img[idy,idx])
                    d = ((idx - x)**2 + (idy - y)**2)**.5
                    soma += img[idy,idx]/d
                    soma_dist += 1/d
                    #print(soma,d)
                    break
            if near_found:
                break
            
            
    if x!=0 and y!=N-1:
        #superior direito    
        near_found = False
        for i in range(1,p+1):
            idx = x + i
            if idx > M -1:
                idx = M - 1 
            for j in range(1,p+1):
                idy = y-j
                if idy < 0:
                    idy = 0
                if img[idy,idx] > -1:
                    near_found = True
                    #print(img[idy,idx])
                    d = ((idx - x)**2 + (idy - y)**2)**.5
                    soma += img[idy,idx]/d
                    soma_dist += 1/d
                    #print(soma,d)
                    break
            if near_found:
                break
            
    if x!= M-1 and y!=0:
        #inferior esquerdo
        near_found = False
        for i in range(1,p+1):
            idx = x-i
            if idx < 0:
                idx = 0
            for j in range(1,p+1):
                idy = y+j
                if idy > N-1:
                    idy = N-1
                if img[idy,idx] > -1:
                    near_found = True
                    #print(img[idy,idx])
                    d = ((idx - x)**2 + (idy - y)**2)**.5
                    soma += img[idy,idx]/d
                    soma_dist += 1/d
                    #print(soma,d)
                    break
            if near_found:
                break
            
    if x!=0 and y!=0:
        #inferior direito
        near_found = False
        for i in range(1,p+1):
            idx = x + i
            if idx > M - 1:
                idx = M - 1 
            for j in range(1,p+1):
                idy = y+j
                if idy > N-1:
                    idy = N-1
                if img[idy,idx] > -1:
                    near_found = True
                    #print(img[idy,idx])
                    d = ((idx - x)**2 + (idy - y)**2)**.5
                    soma += img[idy,idx]/d
                    soma_dist += 1/d
                    #print(soma,d)
                    break
            if near_found:
                break

    return (soma/soma_dist)

if __name__ == '__main__':

    img  = array([[1, -1, -1],
                  [5, -1, 9],
                  [6, 2, -1],
                  [8, -1, -1]])
    
    
    valor = find_nearest(img,2,3,1.5)
    print(valor)

