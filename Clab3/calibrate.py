# -*- coding: utf-8 -*-
# CLAB3 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from numpy import save,load
import cv2

I = Image.open('stereo2012a.jpg');
image_new = cv2.imread('stereo2012a.jpg')
cal_image = image_new
plt.imshow(I)
uv = plt.ginput(6) # Graphical user interface to get 6 points

#####################################################################
def calibrate(im, XYZ, uv):
    # TBD
    XYZ_new = np.asarray(XYZ)
    uv_new = np.asarray(uv)
    variable_N = XYZ_new.shape[0]
    new_array_formed = []
    
    if (variable_N <= 5):
        print("Please enter the more calibaration points as minimum 6 points are required")
    else:
        for j in range(variable_N):
            x,y,z = XYZ_new[j,0],XYZ_new[j,1],XYZ_new[j,2] # Extracting the values for the  x,y,z
            u,v = uv_new[j,0],uv_new[j,1] # Extracting the values for the u and v
            new_array_formed.append([x,y,z,1,0,0,0,0,-u*x,-u*y,-u*z,-u]) # Appending the list with the values
            new_array_formed.append([0,0,0,0,x,y,z,1,-v*x,-v*y,-v*z,-v]) # Appending the list with the values
            
        new_array_formed = np.asarray(new_array_formed) # Converting the list to the array
    
    S,V,D = np.linalg.svd(new_array_formed) # Conversion of the matrix using the singular value decomposition
    normalizing_q = D[-1,-1]
    C = D[-1,:]/normalizing_q # Noramlizing the values
    C = np.array(C)
    C = C.reshape((3,4)) # Reshaping the array into the desired form
    return C


'''
%% TASK 1: CALIBRATE
%
% Function to perform camera calibration
%
% Usage:   calibrate(image, XYZ, uv)
%          return C
%   Where:   image - is the image of the calibration target.
%            XYZ - is a N x 3 array of  XYZ coordinates
%                  of the calibration target points. 
%            uv  - is a N x 2 array of the image coordinates
%                  of the calibration target points.
%            K   - is the 3 x 4 camera calibration matrix.
%  The variable N should be an integer greater than or equal to 6.
%
%  This function plots the uv coordinates onto the image of the calibration
%  target. 
%
%  It also projects the XYZ coordinates back into image coordinates using
%  the calibration matrix and plots these points too as 
%  a visual check on the accuracy of the calibration process.
%
%  Lines from the origin to the vanishing points in the X, Y and Z
%  directions are overlaid on the image. 
%
%  The mean squared error between the positions of the uv coordinates 
%  and the projected XYZ coordinates is also reported.
%
%  The function should also report the error in satisfying the 
%  camera calibration matrix constraints.
% 
% your name, date
% Prateek Arora(u6742441), 5th June 2020
'''
new_uv_points = []

## The below two functions helps in decomposing the calibrated matrix into the three matrix named K,R,t
def vgg_rq(S):
    S = S.T
    [Q,U] = np.linalg.qr(S[::-1,::-1], mode='complete')

    Q = Q.T
    Q = Q[::-1, ::-1]
    U = U.T
    U = U[::-1, ::-1]
    if np.linalg.det(Q)<0:
        U[:,0] = -U[:,0]
        Q[0,:] = -Q[0,:]
    return U,Q

def vgg_KR_from_P(P, noscale = True):
    N = P.shape[0]
    H = P[:,0:N]
    print(N,'|', H)
    [K,R] = vgg_rq(H)
    if noscale:
        K = K / K[N-1,N-1]
        if K[0,0] < 0:
            D = np.diag([-1, -1, np.ones([1,N-2])])
            K = K.astype('float')
            D = D.astype('float')
            K = np.matmul(K,D)
            R = D @ R
        
            test = K*R; 
            assert (test/test[0,0] - H/H[0,0]).all() <= 1e-07
    
    t = np.linalg.inv(-P[:,0:N]) @ P[:,-1]
    return K, R, t


XYZ_grid = [[0,7,7,1],[0,7,14,1],[0,7,21,1],[7,7,0,1],[7,14,0,1],[14,0,14,1]] # Grid select for the calibration

save('XYZ_grid.npy',XYZ_grid) # Saving the xyz coordinates

XYZ_data_extracted = load('XYZ_grid.npy') # Extracting the xyz coordinates

save('uv_array.npy',uv) #Saving the uv coordinates

uv_value_extracted = load('uv_array.npy') #Extracting the uv coordinates

caliberated = calibrate(I,XYZ_data_extracted,uv_value_extracted) # Storing the calibrated coordinates

print("Calibrated matrix", caliberated)

K_value,R_value,t_value = vgg_KR_from_P(caliberated)

print("Value of K is", K_value) # Printing the K matrix
print("Value of R is", R_value) # Printing the R matrix
print("Value of t is", t_value) # Printing the t matrix

new_uv_points = []

new_uv_points = np.dot(XYZ_data_extracted,caliberated.T) # Extracting the new uv values

for k in range(len(uv_value_extracted)):
    new_u,new_v = uv_value_extracted[k]
    new_v = int(new_v)
    new_u = int(new_u)
    image_new[new_v-3:new_v+3, new_u-3:new_u+3] = [0,225,0] # Making the color green which shows the point selected in the xyz direction
    
cv2.imshow("Marked points",image_new)
cv2.waitKey(0)
plt.show()
plt.clf()

new_uv_matrix_formed = []

for i in range(len(uv_value_extracted)):
    caliberated_u,caliberated_v,w = new_uv_points[i]
    caliberated_v = int(caliberated_v/w)
    caliberated_u = int(caliberated_u/w)
    new_uv_matrix_formed.append((caliberated_u,caliberated_v))
    cal_image[caliberated_v-3:caliberated_v+3, caliberated_u-3:caliberated_u+3] = [0,0,255] # Making the color red which shows the uv points extracted from the calibrated matrix

new_uv_matrix_formed = np.asarray(new_uv_matrix_formed)

new_uv_matrix_formed = new_uv_matrix_formed.reshape((6,2))

error = np.sum((uv_value_extracted - new_uv_matrix_formed)**2)/6  #Calculating the mean squared error
print("MSE error", error)

cv2.imshow("Caliberated points",cal_image)
cv2.waitKey(0)
plt.show()

