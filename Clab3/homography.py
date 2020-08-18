# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import save,load
import cv2
#
I_base = Image.open('Right.jpg');
I_right = cv2.imread('Right.jpg');
plt.imshow(I_base)
uv = plt.ginput(6)

I_trans = Image.open('Left.jpg');
I_left = cv2.imread('Left.jpg');
plt.imshow(I_trans)
uv1 = plt.ginput(6) # Graphical user interface to get 6 points

'''
%% TASK 2: 
% Computes the homography H applying the Direct Linear Transformation 
% The transformation is such that 
% p = np.matmul(H, p.T), i.e.,
% (uBase, vBase, 1).T = np.matmul(H, (u2Trans , v2Trans, 1).T)
% Note: we assume (a, b, c) => np.concatenate((a, b, c), axis), be careful when 
% deal the value of axis 
%
% INPUTS: 
% u2Trans, v2Trans - vectors with coordinates u and v of the transformed image point (p') 
% uBase, vBase - vectors with coordinates u and v of the original base image point p  
% 
% OUTPUT 
% H - a 3x3 Homography matrix  
% 
% your name, date
% Prateek Arora(u6742441), 5th June 2020
'''

def homography(u2Trans, v2Trans, uBase, vBase):
    
    new_H_array = []
    variable_n = 6
    
    for i in range(variable_n):
        new_H_array.append([0,0,0,-u2Trans[i],-v2Trans[i],-1,vBase[i]*u2Trans[i], vBase[i]*v2Trans[i], vBase[i]]) #Appending the first line of the array required
        new_H_array.append([u2Trans[i],v2Trans[i],1,0,0,0,-uBase[i]*u2Trans[i],-uBase[i]*v2Trans[i],-uBase[i]]) #Appending the second line of the array required
            
    new_H_array = np.asarray(new_H_array) # Converting the list to the array
    S,V,D = np.linalg.svd(new_H_array) #Getting the values using single value decomposition
    normalizing = D[-1,-1] #  Normalizing the values
    H = D[-1,:]/normalizing ## Getting the H 
    H = np.asarray(H) # Convert H to an array
    H = H.reshape((3,3)) # Rehaping the H array to the required shape
    return H


n_ginput_variable = 6 # number of ginput provided

save('uvBase.npy',uv) # Saving the graphical input of "Right" picture

uv_data_extracted = load('uvBase.npy') # Extracting the values

save('uvTrans.npy',uv1) # Saving the graphical input of "Left" picture

uv1_value_extracted = load('uvTrans.npy') #Extracting the values

uv_data_extracted = np.asarray(uv_data_extracted)
uv1_value_extracted = np.asarray(uv1_value_extracted)

uBase, vBase = uv_data_extracted[:,0], uv_data_extracted[:,1] # Getting the value for the uBase and vBase
uTrans, vTrans = uv1_value_extracted[:,0], uv1_value_extracted[:,1] #Getting the value for the vTrans and uTrans

print("uBase", uBase);
print("vBase", vBase);
print("uTrans", uTrans);
print("vTrans", vTrans);

for k in range(n_ginput_variable):
    uu = int(uBase[k])
    vv = int(vBase[k])
    uu1 = int(uTrans[k])
    vv1 = int(vTrans[k])
    I_right[vv-3:vv+3,uu-3:uu+3] = [255,0,0] # Assigning the red color to the selected points
    I_left[vv1-3:vv1+3,uu1-3:uu1+3] = [255,0,0] # Assigning the red color value to the selected points on the another image

plt.imsave("Right_image_after_ginput.jpg",I_right) #Saving the image
plt.imsave("Left_image_after_ginput.jpg",I_left) #Saving the image

Homography_matrix = homography(uTrans,vTrans,uBase,vBase) #Storing the homography matrix after retriving into the variable

print(Homography_matrix)

warp_affine_im = cv2.warpPerspective(I_left, Homography_matrix, (450,450)) #Getting the warped image
plt.imshow(warp_affine_im)# Display of the warp image
plt.show()
