import os 
import numpy as np 
import matplotlib.pyplot as plt

def compare_list_ndarray():
    list_1=[1,"asdfghj,3,4",5,6]
    list_2=[2,"asdfawdawdghj,3,4",15,26]  
    print(list_1+list_2)
    
    list_1=[1,2,3,4]
    list_2=[1,2,3,4]
    list_1+list_2+[10]
    
    print(list_1+list_2+[10])
    list_3=np.asarray([1,2,3,4])
    list_4=np.asarray([1,2,3,4])
    print(list_3+list_4+10)
def get_jpeg_files():
    os.getcwd()
    os.listdir()
    path=os.getcwd()
    jpg_files=[f for f in os.listdir(path) if f.endswith('.jpg')]
    return jpg_files

get_jpeg_files()
compare_list_ndarray()
get_jpeg_files()  
def display_two_image(im_1,im_2):
    plt.subplot(1,2,1)
    plt.imshow(im_1)
    
    plt.subplot(1,2,2)
    plt.imshow(im_2+30)
    
    plt.show
    
def rotate(im_1):
    m,n,k=im_1.shape
    new_image=np.zeros((n,m,k),dtype='uint8')
    
    for i in range(m):
        for j in range(n):
            temp=image_1[i,j]
            new_image[j,i]=temp
            
    return new_image   
    
    
image_1=plt.imread('deneme.jpg')
image_2=rotate(image_1)
display_two_image(image_1,image_2)
