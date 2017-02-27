# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 21:27:56 2017

@author: SROY
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import timeit

def generateMask(img):

    kernel = np.ones((5,5),np.uint8)                            #Kernel function
    
    Gausblur = cv2.GaussianBlur(img,(3,3),0)                    #Gaussian Blur to remove noise
    
    edges = cv2.Canny(Gausblur, 10, 5)                          #Edge detection using canny
    
    ret, threshBin = cv2.threshold(edges, 127, 255,cv2.THRESH_BINARY_INV)     #Binary image
    
    erosionImg = cv2.erode(threshBin,kernel,iterations = 2)     #Erode Image
     
    dilateImg = cv2.dilate(erosionImg,kernel,iterations = 3)    #Dilate Image 
    
    image, contours, hierarchy=cv2.findContours(dilateImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)     #Find Contours
    
    cv2.drawContours(dilateImg, contours, -1, (0, 255, 0), 25)        #Draw Contour
    
    opening = cv2.morphologyEx(dilateImg, cv2.MORPH_OPEN, kernel)     #Remove false pos and false neg
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)    
    mask = 255 - closing                                              #Invert Image
    
    return mask

def findWeightAv(image_files):
    first = os.path.join(folder, image_files[0])    #Read any image to create variable structure
    img = cv2.imread(first, cv2.IMREAD_GRAYSCALE)   #Read Image 
    msk_sum = np.zeros_like(img, dtype=np.float64)  #Create numpy array variable of image structure
    
    for name in image_files:
        image_file = os.path.join(folder, name)             
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        
        genMask = generateMask(img)                 #Create mask for every image in folders
        msk_sum += genMask                          
        
    mean = msk_sum / len(image_files)               #Create an average mask of all image in a folder
    return mean

def main(folder):
    image_files = os.listdir(folder)
    
    start = timeit.timeit()
    mean = findWeightAv(image_files)                #Returns the mean of all images
    end = timeit.timeit()

    print("Time taken : ", end - start)
    
    plt.imshow(mean, cmap="gray")
    plt.title("Final Mask")

#par_dir = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\DataHw1\\sample_drive\\'
#dirs = [d for d in os.listdir(par_dir) if os.path.isdir(os.path.join(par_dir, d))]
#
#for i in range(len(dirs)):
#    folder = par_dir + 'cam_' + str(i)
#    print("Masking Camera "+ str(i) + "... " + folder)
#    #if __name__ == '__main__':   
#    main(folder)
#    print("Masking Camera "+ str(i) + "... complete")

folder = 'C:\\Users\\SROY\\Desktop\\Courses\\CS513\\DataHw1\\sample_drive\\cam_test'
main(folder)