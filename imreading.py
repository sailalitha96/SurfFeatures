# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 23:33:36 2019

@author: Sailalitha
"""
#Adapted from pYopen cv tutorlas to form an basic understanding of the 
#SURF and sift mechanisms for detection
import cv2
import numpy as np
 
img = cv2.imread("the_book_thief.jpg", cv2.IMREAD_GRAYSCALE)
 
sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
 
orb = cv2.ORB_create(nfeatures=1500)
 
keypoints, descriptors = orb.detectAndCompute(img, None)
 
img = cv2.drawKeypoints(img, keypoints, None)
 
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()