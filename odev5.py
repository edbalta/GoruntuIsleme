import os
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
os.getcwd(),os.listdir()

def m_f_0_and(l1, l2):
    n = len(l1)
    s = []
    for i in range (n):
        a = (list1[i] and list2[i])
        s.append(a)
    return s

def m_f_1_AND_or_OR(l1, operator = 0):
    if operator:
        if 1 in l1:
            s1 = 1
        else:
            s1 = 0
    else:
        if 0 in l1:
            s1 = 0
        else:
            s1 = 1
    return s1

def m_f_2_Combine(l1, l2, op = 0):
    a = m_f_0_and(l1, l2)
    return m_f_1_AND_or_OR(a, op)
	

list1 = [0, 0, 1, 0, 1]
list2 = [1, 1, 1, 1, 1]
m_f_2_Combine(list1, list2, 1)

def convert_RGB_to_monochrome_BW(image1, threshold = 100):
    img_1 = image1      #plt.imread(image1)
    img_2 = np.zeros((img_1.shape[0], img_1.shape[1]))
    for i in range(img_2.shape[0]):
        for j in range(img_2.shape[1]):
            if(img_1[i, j, 0]/3 + img_1[i, j, 1]/3 + img_1[i, j, 1]/3) > threshold:
                img_2[i, j] = 0
            else:
                img_2[i, j] = 1
    return img_2





path_file = "E:\ödevler\Goruntu Isleme\odev5\letter.png"
img_1 = plt.imread(path_file)
img_2 = convert_RGB_to_monochrome_BW(img_1, 0.5)

plt.subplot(1, 2, 1), plt.imshow(img_1)
plt.subplot(1, 2, 2), plt.imshow(img_2, cmap = 'gray')
plt.show()

img_1.shape

np.max(img_1)

img_2


def define_mask_1():
    mask_1 = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ###  mask, mask[1][2], mask[0][0], mask[2][2]
    ###  for i in range(3):
    ###      for j in range(3):
    ###          print(mask[i][j], end = " ")
    ###      print()
    return mask_1


def defineMask2():
    mask_1 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    mask, mask[1][2], mask[0][0], mask[2][2]
    for i in range(3):
        for j in range(3):
            print(mask[i][j], end = " ")
        print()
    return mask_1


def my_dilation(img_1, mask):
    m = img_1.shape[0]
    n = img_1.shape[1]
    
    img_2 = np.zeros((m,n),dtype="unit8")
	
    for i in range(1, m-1):
        for j in range(1, n-1):
            
            x1 = img_1[i, j] == mask[1][1]
            
            x2 = img_1[i-1, j-1] == mask[0][0]
            x3 = img_1[i-1, j] == mask[0][1]
            x4 = img_1[i-1, j+1] == mask[0][2]
            
            x5 = img_1[i+1, j-1] == mask[2][0]
            x6 = img_1[i+1, j] == mask[2][1]
            x7 = img_1[i+1, j+1] == mask[2][2]
            
            x8 = img_1[i, j-1] == mask[1][0]
            x9 = img_1[i, j+1] == mask[1][2]
            
            result1 = x1 or x2 or x3 or x4 or x5
            result2 = x6 or x7 or x8 or x9
            
            result = result1 or result2
            
            img_2[i, j] = result
    return img_2

img3 = my_dilation(img_2, define_mask_1())

plt.figure(figsize = (15,15))
plt.subplot(1, 3, 1), plt.imshow(img_1)
plt.subplot(1, 3, 2), plt.imshow(img_2, cmap = 'gray')
plt.subplot(1, 3, 3), plt.imshow(img3, cmap = 'gray')
plt.show()


img4 = my_dilation(img3, define_mask_1())
img5 = my_dilation(img4, define_mask_1())
img6 = my_dilation(img5, define_mask_1())
img7 = my_dilation(img6, define_mask_1())
plt.figure(figsize = (15,15))
plt.subplot(1, 2, 1), plt.imshow(img6, cmap = 'gray')
plt.subplot(1, 2, 2), plt.imshow(img7, cmap = 'gray')
plt.show()	
	
