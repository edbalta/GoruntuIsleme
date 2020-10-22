import os
import numpy as np
import matplotlib.pyplot as plt

def get_value_from_triple(temp1):
    return int(temp1[0]/3+temp1[1]/3+temp1[2]/3)
    #return int(temp1[0]/2+temp1[1]/2+temp1[2]/2)
    
def get_0_1_from_triple(temp1):
    temp=int(temp1[0]/3+temp1[1]/3+temp1[2]/3)
    if temp<110:
        return 0
    else:
        return 1
        
def convert_rgb_to_bw(im_1):
    m,n,k=im_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_0_1_from_triple(im_1[i,j,:])
            new_image[i,j]=s
        
    return new_image

def convert_rgb_to_gray(im_1):
    m,n,k=im_1.shape
    new_image=np.zeros((m,n),dtype='uint8')
    for i in range(m):
        for j in range(n):
            s=get_value_from_triple(im_1[i,j,:])
            new_image[i,j]=s
        
    return new_image
    
    im_1=plt.imread('deneme.jpg')
    
    
im_1_gray=convert_rgb_to_gray(im_1)
im_1_bw=convert_rgb_to_bw(im_1)

plt.imsave("deneme_gray.jpg",im_1_gray,cmap="gray")
plt.imsave("deneme_bw.jpg",im_1_bw,cmap="gray")



plt.subplot(1,3,1)
plt.imshow(im_1)

plt.subplot(1,3,2)
plt.imshow(im_1_gray,cmap="gray")

plt.subplot(1,3,3)
plt.imshow(im_1_bw,cmap="gray")

plt.show()
