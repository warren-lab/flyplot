import sys
import os
import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
## the find fly angle was initially developed by the Dickson Lab at Caltech but is undergoing refinement by the Warren Lab
from find_fly_angle.find_fly_anglev2 import *
def get_objectmask(maskFly,max_contour):
    """
    function that converts the mask of the fly and image to just the
    mask of the fly on a background
    """
    # get the mask of fly that is same size but zeros
    gen_mask = np.zeros(maskFly.shape,dtype='uint8')
    # convert image to BGR
    img_fly_mask_cont = cv2.cvtColor(gen_mask,cv2.COLOR_GRAY2BGR)
    
    # draw the contours on this mask
    cv2.drawContours(img_fly_mask_cont,[max_contour],contourIdx =0, color = (255,255,255),thickness = 4)
    return img_fly_mask_cont

def get_fill_flymask(maskFly, max_contour):
    """
    show the mask fly with contour areay filled in
    """
    # get fly with contour on background
    cont_mask = get_objectmask(maskFly,max_contour)
    # convert back to grayscale
    cont_mask = cv2.cvtColor(cont_mask,cv2.COLOR_BGR2GRAY)
    # fill in the fly contour to get the mask
    cv2.fillPoly(cont_mask,[max_contour],color = (255))
    return cont_mask

def get_contour_centroid(img):
    """
    This function takes an input of an image and returns the contour, centroid and centerline

    """
    img_dict = {
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None
    }
    # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform otsu thresholding
    ### set fly_mask to be img_otsu if wanting to add in a morphological operation
    rval, fly_mask= cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)

    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    ## MORPHOLOCICAL OPERATION IN ORDER TO INCORPORATE SMOOTHING!
    # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])
    print(body_axis_pt_0)
    return fly_mask, max_contour,centroid, body_axis_pt_0,body_axis_pt_1

def contour_hom_matrix(img):
    """
    This function takes input of an image and gets its contour
    
    After this point the following homogenous transform is performed doing a rotation in 2D space 
    
    mat = [
            cos(theta)  -sin(theta)  Cx
            sin(theta)  -cos(theta)  Cy
                0            0        1
          ]
    After obtainin this matrix we get the inverse

    Using this inverted form in order to get the correct value... or use linalgsolv...


    """
def contour_rot_matrix(img):
    """
    This function takes input of an image and gets its contour

    After this point the rotational matrix is performed in order to determine what the original non rotated position would be with respect to the centroid
    """
     # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform thresholding
    otsu_th, img_th = cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)
    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  
    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an morphological operation to smooth the image and then get the mask
    fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # Compute circle mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid

    # Circular Mask
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2

    # ROTATION
    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)
    rotation_mat = np.array([
    [np.cos(angle+np.pi/2), -np.sin(angle+np.pi/2)],
    [np.sin(angle+np.pi/2),  np.cos(angle+np.pi/2)]
])

    rotation_mat =  rotation_mat
    print("YO")
    print(type(rot_mat), rot_mat.shape)
    print(np.rad2deg(angle),90-np.rad2deg(angle),180-np.rad2deg(angle) )
    
    max_contour_adj = max_contour.reshape(-1,2)
    print(np.array([[mid_x - Cx,mid_y-Cy]]).dtype, max_contour_adj.dtype)
    max_contour_adj-=np.array([[Cx,Cy]]).astype(np.int32)
    print(np.shape(max_contour_adj),np.shape(np.ones((max_contour_adj.shape[0],1))))
    # arr = (np.hstack((max_contour_adj,np.ones((max_contour_adj.shape[0],1)))))
    # arr_t =  np.transpose(arr)
    print(rot_mat.shape, rotation_mat.shape,max_contour_adj.shape)
    # unrot_vec = np.linalg.solve(max_contour_adj, rotation_mat) # get the unrotated vector that has been translated
    unrot_vec= np.dot(max_contour_adj,rotation_mat)
    print("Unrotated")
    # print(unrot_vec.shape)
    unrot_vec = unrot_vec.reshape(unrot_vec.shape[0],1,unrot_vec.shape[1])
    # print(unrot_vec.shape)
    # Shift by the centroid
    print(unrot_vec[0])
    print("\ncoords")
    print((mid_x,Cx),(mid_y,Cy))
    unrot_vec+=np.array([[(Cx),(Cy)]])
    unrot_vec = unrot_vec.astype(np.int32)
    print("\nApplied")
    print(unrot_vec[0])

    # Rotation Adjusted Body Axis
    print(body_axis_pt_0,"old body axis")
    # body_axis_pt_0 = np.array(body_axis_pt_0)
    body_axis_pt_0-= np.array([[Cx,Cy]]).astype(int)
    print(rotation_mat.shape, np.array(body_axis_pt_0).shape,body_axis_pt_0.reshape(2,1) )
    body_axis_pt_0 = np.dot(rotation_mat, np.array(body_axis_pt_0).reshape(2,1))
    body_axis_pt_0+= np.array([[Cx,Cy]]).astype(int).reshape(2,1)
    body_axis_pt_0 = body_axis_pt_0.astype(int)
    print(body_axis_pt_0,"new body axis 0")
    print(body_axis_pt_1,"old body axis 1")
    # body_axis_pt_1 = np.array(body_axis_pt_1)
    body_axis_pt_1-= np.array([[Cx,Cy]]).astype(int)
    print(rotation_mat.shape, np.array(body_axis_pt_1).shape,body_axis_pt_1.reshape(2,1) )
    body_axis_pt_1 = np.dot(rotation_mat, np.array(body_axis_pt_1).reshape(2,1))
    body_axis_pt_1+= np.array([[Cx,Cy]]).astype(int).reshape(2,1)
    body_axis_pt_1 = body_axis_pt_1.astype(int)
    print(body_axis_pt_1,"new body axis 1")
    print("ANGLE:",angle,np.rad2deg(angle))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(unrot_vec))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])



    return fly_mask, max_contour,centroid, body_axis_pt_0,body_axis_pt_1,unrot_vec

def contour_rot_matrix_shift(img):
    """
    WORKKING VERSION -> 5/9/2024
    This function performs the rotation of the fly regardless of any angle that it is at and get it such that its centerline is aligned with the y-axis.

    This involved using a rotational matrix, and then also 
    """
     # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform thresholding
    otsu_th, img_th = cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)
    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  
    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an morphological operation to smooth the image and then get the mask
    fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # Compute circle mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid

    # Circular Mask
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2

    # ROTATION
    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)
    rotation_mat = np.array([
    [np.cos(angle+np.pi/2), -np.sin(angle+np.pi/2)],
    [np.sin(angle+np.pi/2),  np.cos(angle+np.pi/2)]
])
    # Difference relative to centroid
    x_diff = mid_x - Cx
    y_diff = mid_y - Cy
    rotation_mat =  rotation_mat
    print(type(rot_mat), rot_mat.shape)
    print(np.rad2deg(angle),90-np.rad2deg(angle),180-np.rad2deg(angle) )
    
    max_contour_adj = max_contour.reshape(-1,2)
    print(np.array([[mid_x - Cx,mid_y-Cy]]).dtype, max_contour_adj.dtype)
    max_contour_adj-=np.array([[Cx,Cy]]).astype(np.int32)
    print(np.shape(max_contour_adj),np.shape(np.ones((max_contour_adj.shape[0],1))))
    # arr = (np.hstack((max_contour_adj,np.ones((max_contour_adj.shape[0],1)))))
    # arr_t =  np.transpose(arr)
    print(rot_mat.shape, rotation_mat.shape,max_contour_adj.shape)
    # unrot_vec = np.linalg.solve(max_contour_adj, rotation_mat) # get the unrotated vector that has been translated
    unrot_vec= np.dot(max_contour_adj,rotation_mat)
    print("Unrotated")
    # print(unrot_vec.shape)
    unrot_vec = unrot_vec.reshape(unrot_vec.shape[0],1,unrot_vec.shape[1])
    # print(unrot_vec.shape)
    # Shift by the centroid
    print(unrot_vec[0])
    print("\ncoords")
    print((mid_x,Cx, mid_x-Cx),(mid_y,Cy, mid_y - Cy))
    unrot_vec+=np.array([[int(Cx+x_diff),int(Cy+y_diff)]])
    unrot_vec = unrot_vec.astype(np.int32)
    # print("\nApplied")
    # print(unrot_vec[0])

    # Cx, Cy = (int(mid_x), int(mid_y))
    # Rotation Adjusted Body Axis
    # print(body_axis_pt_0,"old body axis")
    # body_axis_pt_0 = np.array(body_axis_pt_0)
    # body_axis_pt_0 -= np.array([[centroid[0],centroid[1]]]).astype(int)
    # print(rotation_mat.shape, np.array(body_axis_pt_0).shape,body_axis_pt_0.reshape(2,1) )
    # body_axis_pt_0 = np.dot(rotation_mat, np.array(body_axis_pt_0).reshape(2,1))
    # body_axis_pt_0 += np.array([[centroid[0]+x_diff,centroid[1]+y_diff]]).astype(int).reshape(2,1)
    # body_axis_pt_0 = body_axis_pt_0.astype(int)
    # print(body_axis_pt_0,"new body axis 0")
    # print(body_axis_pt_1,"old body axis 1")
    # # body_axis_pt_1 = np.array(body_axis_pt_1)
    # body_axis_pt_1-= np.array([[centroid[0],centroid[1]]]).astype(int)
    # print(rotation_mat.shape, np.array(body_axis_pt_1).shape,body_axis_pt_1.reshape(2,1) )
    # body_axis_pt_1 = np.dot(rotation_mat, np.array(body_axis_pt_1).reshape(2,1))
    # body_axis_pt_1 += np.array([[centroid[0]+x_diff,centroid[1]+y_diff]]).astype(int).reshape(2,1)
    # body_axis_pt_1 = body_axis_pt_1.astype(int)
    # print(body_axis_pt_1,"new body axis 1")
    # print("ANGLE:",angle,np.rad2deg(angle))
    # # determine the angle and body vectory using finde_fly_angle function
    # angle, body_vect = get_angle_and_body_vector(cv2.moments(unrot_vec))
    # # Get bounding box and find diagonal - used for drawing body axis
    # bbox = cv2.boundingRect(max_contour)
    # bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # # Create points for drawing axis fly in contours image 
    # axis_length = 0.75*bbox_diag
    # body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    # body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])
    print(body_axis_pt_0)
    body_axis_pt_0 = (int(body_axis_pt_0[0]+x_diff),int(body_axis_pt_0[1]+y_diff))
    body_axis_pt_1 = (int(body_axis_pt_1[0]+x_diff),int(body_axis_pt_1[1]+y_diff))
    centroid = (int(mid_x), int(mid_y))
    # print(body_axis_pt_0[0],body_axis_pt_0[1])
    return fly_mask, max_contour,centroid, body_axis_pt_0,body_axis_pt_1,unrot_vec

################# RESTART NEW METHOD.... ROTATE THE IMAGE THEN GET THE INFORMATION   ##############
def get_contour_rotated(img):
    """
    This function takes an input of an image and gets the contour

    Then proceeds to rotate the contour about the centroid of the image as well as rotate the image itself
    
    returns the contour, centroid and centerline

    """
    # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform thresholding
    otsu_th, img_th = cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)
    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  
    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an morphological operation to smooth the image and then get the mask
    fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_th, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # Compute circle mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
    # Circular Mask
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2

    # ROTATION
    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)
    rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = cv2.rotate(rotated_threshold_image, cv2.ROTATE_90_CLOCKWISE)
    rotated_threshold_image = rotated_threshold_image*circ_mask

    # Rotate the Contour



    # Get orientation discriminant and flip image if needed 
    # orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    # if not orient_ok:
    #     rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
    #     rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
    #     rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
    #     angle += np.deg2rad(-180.0)

    # # Get basic image data
    # height, width = rotated_threshold_image.shape
    # print("height",height)
    # image_cvsize = width, height 
    # mid_x, mid_y = 0.5*width, 0.5*height
    # # perform otsu thresholding
    # otsu_th, fly_mask = cv2.threshold(rotated_threshold_image,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)


    # # (Optional) -> Morphology: erosion
    # strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # # get the contours
    # contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # # utilize find_fly_angle function to get max area and contour
    # max_contour, max_area = get_max_area_contour(contours)
    # # determine the centroid using find_fly_angle function
    # Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    # ## centroid
    # centroid = (int(Cx), int(Cy))
    # # determine the angle and body vectory using finde_fly_angle function
    # angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # # Get bounding box and find diagonal - used for drawing body axis
    # bbox = cv2.boundingRect(max_contour)
    # bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # # Create points for drawing axis fly in contours image 
    # axis_length = 0.75*bbox_diag
    # body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    # body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # On the rotated image

    # # Shift and rotate threshold image. 
    # shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    # rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    # rotated_threshold_image = rotated_threshold_image*circ_mask
    # rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    # # Get orientation discriminant and flip image if needed 
    # orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    # if not orient_ok:
    #     rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
    #     rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
    #     rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
    #     angle += np.deg2rad(-180.0)
    
    return rotated_threshold_image, max_contour,centroid, body_axis_pt_0,body_axis_pt_1


def get_contour_centroid_adj(img):
    """
    This function takes an input of an image and 

    rotates this image 
    
    returns the contour, centroid and centerline

    """
    img_dict = {
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None
    }
    # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform otsu thresholding
    otsu_th, img_otsu = cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)


    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # Compute cirlce mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
    # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

    ## plot arrow of direction
    endpoint_x = centroid[0] + axis_length*body_vect[0]
    endpoint_y = centroid[1] - axis_length*body_vect[1]
    # plt.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
    #             width=1,  # Width of the arrow
    #             head_width=30,  # Width of the arrow head
    #             head_length=35,  # Length of the arrow head
    #             fc='blue',  # Fill color of the arrow
    #             ec='blue',  # Edge color of the arrow
    #             linestyle='-',  # Style of the arrow
    #             length_includes_head=True

    #             )
    # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

    # ##  plot the image
    # plt.imshow(img_fly,cmap ='gray')
    
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)
    rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = cv2.rotate(rotated_threshold_image, cv2.ROTATE_90_CLOCKWISE)
    rotated_threshold_image = rotated_threshold_image*circ_mask
    # Get orientation discriminant and flip image if needed 
    # orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    # if not orient_ok:
    #     rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
    #     rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
    #     rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
    #     angle += np.deg2rad(-180.0)

    # Get basic image data
    height, width = rotated_threshold_image.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform otsu thresholding
    otsu_th, fly_mask = cv2.threshold(rotated_threshold_image,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)


    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # On the rotated image

    # # Shift and rotate threshold image. 
    # shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    # rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    # rotated_threshold_image = rotated_threshold_image*circ_mask
    # rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    # # Get orientation discriminant and flip image if needed 
    # orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    # if not orient_ok:
    #     rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
    #     rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
    #     rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
    #     angle += np.deg2rad(-180.0)
    
    return fly_mask, max_contour,centroid, body_axis_pt_0,body_axis_pt_1


    # # Compute cirlce mask
    # mask_radius = int(.95*height/2.0)
    # print('mask radius',mask_radius)
    # vals_x = np.arange(0.0,width)
    # vals_y = np.arange(0.0,height)
    # grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
    # # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

    # ## plot arrow of direction
    # endpoint_x = centroid[0] + axis_length*body_vect[0]
    # endpoint_y = centroid[1] - axis_length*body_vect[1]
    # # plmalet.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
    # #             width=1,  # Width of the arrow
    # #             head_width=30,  # Width of the arrow head
    # #             head_length=35,  # Length of the arrow head
    # #             fc='blue',  # Fill color of the arrow
    # #             ec='blue',  # Edge color of the arrow
    # #             linestyle='-',  # Style of the arrow
    # #             length_includes_head=True

    # #             )
    # # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

    # # ##  plot the image
    # # plt.imshow(img_fly,cmap ='gray')
    
    # circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



    # # Get matrices for shifting (centering) and rotating the image
    # shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    # rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # # Shift and rotate the original image
    # shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
    # rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

    # # Shift and rotate threshold image. 
    # shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    # rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    # rotated_threshold_image = rotated_threshold_image*circ_mask
    # rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    # # Get orientation discriminant and flip image if needed 
    # orient_ok, orient_discrim= is_orientation_ok(rotated_threshold_image,2)
    # if not orient_ok:
    #     rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
    #     rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
    #     rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
    #     angle += np.deg2rad(-180.0)

    # # angle = normalize_angle_range(angle)
    # # print("angle final", angle)        fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

    # # The images will be kept in grayscale
    # # Correct contours
    # max_cont_fix = max_contour.reshape(-1,2)
    # ## plot the contour
    # # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)

    # # Here we will save the information to the corresponding dictionary key
    # fly_gender = type_list[idx]
    # img_dict[fly_gender]['img'] = img_fly
    # img_dict[fly_gender]['centroid'] = centroid
    # img_dict[fly_gender]['body_axis_pt_0']= body_axis_pt_0
    # img_dict[fly_gender]['body_axis_pt_1']= body_axis_pt_1
    # img_dict[fly_gender]['max contour']= max_cont_fix




# if display == True:
#     # print(max_cont_fix)
#     plt.figure(figsize=(24,24))
#     ## will create 4 image panels
#     ### 1 the original male image
#     plt.subplot(2,2,1)
#     plt.title(f"Original {fly1} Image", fontsize = 20)
#     plt.axis('off')
#     plt.imshow(img_dict['male']['img'],cmap = 'gray')
#     # ### 2 optional show the image mask
#     # plt.subplot(2,2,2)
#     # plt.tiangletle("Fly Mask", fontsize = 10)
#     # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
#     # plt.imshow(masked_filled_fly,cmap ='gray')
#     ### 3 original male image with the contour and line    ## plot the centroid
#     plt.subplot(2,2,2)
#     plt.title(f"{fly1} Fly Contour and Orientation", fontsize = 20)
    
#     plt.plot(img_dict['male']['centroid'][0],img_dict['male']['centroid'][1],marker = '.',color = 'r' ,markersize= 10)
#     ## plot the contour
#     plt.plot(img_dict['male']['max contour'][:,0],img_dict['male']['max contour'][:,1],color = 'r', linewidth = 2)
#     # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
#     print(img_dict['male']['body_axis_pt_0'])
#     plt.plot((img_dict['male']['body_axis_pt_0'][0],img_dict['male']['body_axis_pt_1'][0]),(img_dict['male']['body_axis_pt_0'][1],img_dict['male']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
#     # cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
#     plt.axis('off')
#     plt.imshow(img_dict['male']['img'],cmap = 'gray')
    
#     ### 3 the original female image
#     plt.subplot(2,2,3)
#     plt.title(f"Original {fly2} Image", fontsize = 20)
#     plt.axis('off')
#     plt.imshow(img_dict['female']['img'],cmap = 'gray')
#     # ### 2 optional show the image mask
#     # plt.subplot(2,2,2)
#     # plt.tiangletle("Fly Mask", fontsize = 10)
#     # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
#     # plt.imshow(masked_filled_fly,cmap ='gray')
#     ### 4 original female image with the contour and line    ## plot the centroid
#     plt.subplot(2,2,4)
#     plt.title(f"{fly2} Fly Contour and Orientation", fontsize = 20)
#     ## plot the contour
#     plt.plot(img_dict['female']['max contour'][:,0],img_dict['female']['max contour'][:,1],color = 'r', linewidth = 2)
#     ## plot the centerline
#     plt.plot((img_dict['female']['body_axis_pt_0'][0],img_dict['female']['body_axis_pt_1'][0]),(img_dict['female']['body_axis_pt_0'][1],img_dict['female']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
#     ## get rid of the axis
#     plt.axis('off')
#     ##  plot the image
#     plt.imshow(img_fly,cmap ='gray')
#     plt.savefig('figs/2by2_orig_thr.svg',facecolor = 'white')

## single image
def single_image_save(file_name,display=False):
    """
    single image save function that takes known file name.
    the user can additionally allow for the images to be displayed within the notebook

    Some of the code within this function is taken from the find_fly_angle module
    """
    # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(file_name,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform otsu thresholding
    otsu_th, img_otsu = cv2.threshold(img_fly,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)

    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])

    # Compute cirlce mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
    # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

    ## plot arrow of direction
    endpoint_x = centroid[0] + axis_length*body_vect[0]
    endpoint_y = centroid[1] - axis_length*body_vect[1]
    # plt.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
    #             width=1,  # Width of the arrow
    #             head_width=30,  # Width of the arrow head
    #             head_length=35,  # Length of the arrow head
    #             fc='blue',  # Fill color of the arrow
    #             ec='blue',  # Edge color of the arrow
    #             linestyle='-',  # Style of the arrow
    #             length_includes_head=True

    #             )
    # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

    # ##  plot the image
    # plt.imshow(img_fly,cmap ='gray')
    
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = rotated_threshold_image*circ_mask
    rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    # Get orientation discriminant and flip image if needed 
    orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    if not orient_ok:
        rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
        rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
        rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
        angle += np.deg2rad(-180.0)

    # angle = normalize_angle_range(angle)
    print("angle final", angle)
    # The images will be kept in grayscale
    # Correct contours
    max_cont_fix = max_contour.reshape(-1,2)
    ## plot the contour
    # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)


    if display == True:
        print(max_cont_fix)
        plt.figure(figsize=(24,24))
        ## will create 4 image panels
        ### 1 the original image
        plt.subplot(2,2,1)
        plt.title("Original Image", fontsize=10)
        plt.imshow(img_fly,cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.title("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 3 original image with the contour and line    ## plot the centroid
        plt.subplot(2,2,2)
        plt.title("Fly Contour and Orientation")
        
        plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)
        ## plot the contour
        plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)
        ## plot arrow of direction
        # endpoint_x = centroid[0] + axis_length*body_vect[0]
        # endpoint_y = centroid[1] - axis_length*body_vect[1]
        # plt.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
        #             width=1,  # Width of the arrow
        #             head_width=30,  # Width of the arrow head
        #             head_length=35,  # Length of the arrow head
        #             fc='blue',  # Fill color of the arrow
        #             ec='blue',  # Edge color of the arrow
        #             linestyle='-',  # Style of the arrow
        #             length_includes_head=True

        #             )
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
        plt.plot((body_axis_pt_0[0],body_axis_pt_1[0]),(body_axis_pt_0[1],body_axis_pt_1[1]), color = 'r')
        cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
        ##  plot the image
        plt.imshow(img_fly,cmap ='gray')

        plt.subplot(2,2,3)
        plt.imshow(rotated_threshold_image, cmap ='gray')


        # Rotated Threshold need to rotate 90 again...
        image = cv2.rotate(rotated_image, cv2.ROTATE_90_CLOCKWISE)
        plt.imshow(image, cmap ='gray')

        # plt.savefig('images!')

    # save this figure
    # plt.figure(figsize=(15,15))
    # plt.title("Fly Contour and Orientation")
    # ## plot the centroid
    # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)
    # ## plot the contour
    # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)
    ## plot arrow of direction
    # endpoint_x = centroid[0] + axis_length*body_vect[0]
    # endpoint_y = centroid[1] - axis_length*body_vect[1]
    # plt.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
    #             width=1,  # Width of the arrow        plt.subplot(3,1,3)
        # plt.title("Fly Contour and Orientation")
        # ### 3 original image with the contour and arrow/line    ## plot the centroid
        # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)
        # ## plot the contour
        # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)
        # ## plot arrow of direction
        # # endpoint_x = centroid[0] + axis_length*body_vect[0]
        # # endpoint_y = centroid[1] - axis_length*body_vect[1]
        # # plt.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
        # #             width=1,  # Width of the arrow
        # #             head_width=30,  # Width of the arrow head
        # #             head_length=35,  # Length of the arrow head
        # #             fc='blue',  # Fill color of the arrow
        # #             ec='blue',  # Edge color of the arrow
        # #             linestyle='-',  # Style of the arrow
        # #             length_includes_head=True
        # plmalet.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
        #             width=1,  # Width of the arrow
        #             head_width=30,  # Width of the arrow head
        #             head_length=35,  # Length of the arrow head
        #             fc='blue',  # Fill color of the arrow
        #             ec='blue',  # Edge color of the arrow
        #             linestyle='-',  # Style of the arrow
        #             length_includes_head=True

        #             )
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

        # ##  plot the image
        # plt.imshow(img_fly,cmap ='gray')
        
        circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



        # Shift and rotate threshold image. 
        shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
        rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
        rotated_threshold_image = rotated_threshold_image*circ_mask
        rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
        # Get orientation discriminant and flip image if needed 
        orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
        if not orient_ok:
            rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
            rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
            rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
            angle += np.deg2rad(-180.0)

        # angle = normalize_angle_range(angle)
        # print("angle final", angle)        fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

        # The images will be kept in grayscale
        # Correct contours
        max_cont_fix = max_contour.reshape(-1,2)
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

        # ##  plot the image
        # plt.imshow(img_fly,cmap ='gray')
        # plt.savefig('images!')
    #             head_width=30,  # Width of the arrow head
    #             head_length=35,  # Length of the arrow head
    #             fc='blue',  # Fill color of the arrow
    #             ec='blue',  # Edge color of the arrow
    #             linestyle='-',  # Style of the arrow
    #             length_includes_head=True

    #             )
    # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

    ##  plot the image
    # plt.imshow(img_fly,cmap ='gray')
        ## save the image:
    


## multiple images

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# New method that will do a 2x2 frame for two images, male and female fly
def img_contour_direction(img_file,display=True):
    """
    displays fly image with contour and direction within the notebook

    Some of the code within this function is taken from the find_fly_angle module
    """
    # read in the image file of the fly as Grayscale
    img_fly = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    # Get basic image data
    height, width = img_fly.shape
    print("height",height)
    image_cvsize = width, height 
    mid_x, mid_y = 0.5*width, 0.5*height
    # perform otsu thresholding
    ### set fly_mask to be img_otsu if wanting to add in a morphological operation
    otsu_th, fly_mask= cv2.threshold(img_fly,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)

    # (Optional) -> Morphology: erosion
    strel = np.ones((5,5),np.uint8)  

    # maskFly = cv2.dilate(img_otsu, strel)
    # perform an erosing
    ## MORPHOLOCICAL OPERATION IN ORDER TO INCORPORATE SMOOTHING!
    # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


    # get the contours
    contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    # utilize find_fly_angle function to get max area and contour
    max_contour, max_area = get_max_area_contour(contours)
    # determine the centroid using find_fly_angle function
    Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
    ## centroid
    centroid = (int(Cx), int(Cy))
    # determine the angle and body vectory using finde_fly_angle function
    angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
    # Get bounding box and find diagonal - used for drawing body axis
    bbox = cv2.boundingRect(max_contour)
    bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
    # Create points for drawing axis fly in contours image 
    axis_length = 0.75*bbox_diag
    body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
    body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])
    print(body_axis_pt_0)

    # Compute cirlce mask
    mask_radius = int(.95*height/2.0)
    print('mask radius',mask_radius)
    vals_x = np.arange(0.0,width)
    vals_y = np.arange(0.0,height)
    grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
    # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

    ## plot arrow of direction
    endpoint_x = centroid[0] + axis_length*body_vect[0]
    endpoint_y = centroid[1] - axis_length*body_vect[1]
    # plmalet.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
    #             width=1,  # Width of the arrow
    #             head_width=30,  # Width of the arrow head
    #             head_length=35,  # Length of the arrow head
    #             fc='blue',  # Fill color of the arrow
    #             ec='blue',  # Edge color of the arrow
    #             linestyle='-',  # Style of the arrow
    #             length_includes_head=True

    #             )
    # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

    # ##  plot the image
    # plt.imshow(img_fly,cmap ='gray')
    
    circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



    # Get matrices for shifting (centering) and rotating the image
    shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
    rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

    # Shift and rotate the original image
    shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
    rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

    # Shift and rotate threshold image. 
    shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
    rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
    rotated_threshold_image = rotated_threshold_image*circ_mask
    rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
    # Get orientation discriminant and flip image if needed 
    orient_ok, orient_discrim = is_orientation_ok(rotated_threshold_image,2)
    if not orient_ok:
        rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
        rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
        rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
        angle += np.deg2rad(-180.0)
    # angle = normalize_angle_range(angle)
    # print("angle final", angle)        fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

    # The images will be kept in grayscale
    # Correct contours
    max_cont_fix = max_contour.reshape(-1,2)
    ## plot the contour
    # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)
    if display == True:
        # print(max_cont_fix)
        plt.figure(figsize=(24,12))
        ## will create 4 image panels
        ### 1 the original male image
        plt.subplot(1,2,1)
        plt.title("Original Male Image", fontsize = 20)
        plt.axis('off')
        plt.imshow(img_fly,cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.tiangletle("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 3 original male image with the contour and line    ## plot the centroid
        plt.subplot(1,2,2)
        plt.title("Male Fly Contour and Orientation", fontsize = 20)
        cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
        plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)
        ## plot the contour
        plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
        print(body_axis_pt_0)
        plt.plot((body_axis_pt_0[0],body_axis_pt_1[0]),(body_axis_pt_0[1],body_axis_pt_1[1]), color = 'r', linewidth =2)
        # cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
        plt.axis('off')
        plt.imshow(img_fly,cmap = 'gray')
        plt.show()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Current Method of Thresholding
# New method that will do a 2x2 frame for two images, male and female fly
def two_image_save(male_img_file,fem_img_file,fly1,fly2,display=False):
    """
    two image save function that takes known male file name 
    the user can additionally allow for the images to be displayed within the notebook

    Some of the code within this function is taken from the find_fly_angle module
    """
    img_list = [male_img_file,fem_img_file]
    type_list = ['male','female']
    img_dict = {'male':{
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None
        },
        'female':{
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None,
        'max contour': None
        }
    }
    for idx in range(len(img_list)):
        # read in the image file of the fly as Grayscale
        img_fly = cv2.imread(img_list[idx],cv2.IMREAD_GRAYSCALE)
        # Get basic image data
        height, width = img_fly.shape
        print("height",height)
        image_cvsize = width, height 
        mid_x, mid_y = 0.5*width, 0.5*height
        # perform otsu thresholding
        ### set fly_mask to be img_otsu if wanting to add in a morphological operation
        rval, fly_mask= cv2.threshold(img_fly,25,np.iinfo(img_fly.dtype).max,cv2.THRESH_BINARY_INV)

        # (Optional) -> Morphology: erosion
        strel = np.ones((5,5),np.uint8)  

        # maskFly = cv2.dilate(img_otsu, strel)
        # perform an erosing
        ## MORPHOLOCICAL OPERATION IN ORDER TO INCORPORATE SMOOTHING!
        # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


        # get the contours
        contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # utilize find_fly_angle function to get max area and contour
        max_contour, max_area = get_max_area_contour(contours)
        # determine the centroid using find_fly_angle function
        Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
        ## centroid
        centroid = (int(Cx), int(Cy))
        # determine the angle and body vectory using finde_fly_angle function
        angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
        # Get bounding box and find diagonal - used for drawing body axis
        bbox = cv2.boundingRect(max_contour)
        bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
        # Create points for drawing axis fly in contours image 
        axis_length = 0.75*bbox_diag
        body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
        body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])
        print(body_axis_pt_0)

        # Compute cirlce mask
        mask_radius = int(.95*height/2.0)
        print('mask radius',mask_radius)
        vals_x = np.arange(0.0,width)
        vals_y = np.arange(0.0,height)
        grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
        # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

        ## plot arrow of direction
        endpoint_x = centroid[0] + axis_length*body_vect[0]
        endpoint_y = centroid[1] - axis_length*body_vect[1]
        # plmalet.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
        #             width=1,  # Width of the arrow
        #             head_width=30,  # Width of the arrow head
        #             head_length=35,  # Length of the arrow head
        #             fc='blue',  # Fill color of the arrow
        #             ec='blue',  # Edge color of the arrow
        #             linestyle='-',  # Style of the arrow
        #             length_includes_head=True

        #             )
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

        # ##  plot the image
        # plt.imshow(img_fly,cmap ='gray')
        
        circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



        # Get matrices for shifting (centering) and rotating the image
        shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
        rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

        # Shift and rotate the original image
        shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
        rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

        # Shift and rotate threshold image. 
        shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
        rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
        rotated_threshold_image = rotated_threshold_image*circ_mask
        rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
        # Get orientation discriminant and flip image if needed 
        orient_ok, orient_discrim= is_orientation_ok(rotated_threshold_image,2)
        if not orient_ok:
            rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
            rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
            rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
            angle += np.deg2rad(-180.0)

        # angle = normalize_angle_range(angle)
        # print("angle final", angle)        fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

        # The images will be kept in grayscale
        # Correct contours
        max_cont_fix = max_contour.reshape(-1,2)
        ## plot the contour
        # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)

        # Here we will save the information to the corresponding dictionary key
        fly_gender = type_list[idx]
        img_dict[fly_gender]['img'] = img_fly
        img_dict[fly_gender]['centroid'] = centroid
        img_dict[fly_gender]['body_axis_pt_0']= body_axis_pt_0
        img_dict[fly_gender]['body_axis_pt_1']= body_axis_pt_1
        img_dict[fly_gender]['max contour']= max_cont_fix




    if display == True:
        # print(max_cont_fix)
        plt.figure(figsize=(24,24))
        ## will create 4 image panels
        ### 1 the original male image
        plt.subplot(2,2,1)
        plt.title(f"Original {fly1} Image", fontsize = 20)
        plt.axis('off')
        plt.imshow(img_dict['male']['img'],cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.tiangletle("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 3 original male image with the contour and line    ## plot the centroid
        plt.subplot(2,2,2)
        plt.title(f"{fly1} Fly Contour and Orientation", fontsize = 20)
        
        plt.plot(img_dict['male']['centroid'][0],img_dict['male']['centroid'][1],marker = '.',color = 'r' ,markersize= 10)
        ## plot the contour
        plt.plot(img_dict['male']['max contour'][:,0],img_dict['male']['max contour'][:,1],color = 'r', linewidth = 2)
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
        print(img_dict['male']['body_axis_pt_0'])
        plt.plot((img_dict['male']['body_axis_pt_0'][0],img_dict['male']['body_axis_pt_1'][0]),(img_dict['male']['body_axis_pt_0'][1],img_dict['male']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
        # cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
        plt.axis('off')
        plt.imshow(img_dict['male']['img'],cmap = 'gray')
        
        ### 3 the original female image
        plt.subplot(2,2,3)
        plt.title(f"Original {fly2} Image", fontsize = 20)
        plt.axis('off')
        plt.imshow(img_dict['female']['img'],cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.tiangletle("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 4 original female image with the contour and line    ## plot the centroid
        plt.subplot(2,2,4)
        plt.title(f"{fly2} Fly Contour and Orientation", fontsize = 20)
        ## plot the contour
        plt.plot(img_dict['female']['max contour'][:,0],img_dict['female']['max contour'][:,1],color = 'r', linewidth = 2)
        ## plot the centerline
        plt.plot((img_dict['female']['body_axis_pt_0'][0],img_dict['female']['body_axis_pt_1'][0]),(img_dict['female']['body_axis_pt_0'][1],img_dict['female']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
        ## get rid of the axis
        plt.axis('off')
        ##  plot the image
        plt.imshow(img_fly,cmap ='gray')
        plt.savefig('figs/2by2_orig_thr.svg',facecolor = 'white')

# New method that will do a 2x2 frame for two images, male and female fly
def two_image_save_Otsu(male_img_file,fem_img_file,fly1,fly2,display=False):
    """
    two image save function that takes known male file name 
    the user can additionally allow for the images to be displayed within the notebook

    Some of the code within this function is taken from the find_fly_angle module
    """
    img_list = [male_img_file,fem_img_file]
    type_list = ['male','female']
    img_dict = {'male':{
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None
        },
        'female':{
        'img':None,
        'centroid': None,
        'body_axis_pt_0':None,
        'body_axis_pt_1':None,
        'max contour': None
        }
    }
    for idx in range(len(img_list)):
        # read in the image file of the fly as Grayscale
        img_fly = cv2.imread(img_list[idx],cv2.IMREAD_GRAYSCALE)
        # Get basic image data
        height, width = img_fly.shape
        print("height",height)
        image_cvsize = width, height 
        mid_x, mid_y = 0.5*width, 0.5*height
        # perform otsu thresholding
        ### set fly_mask to be img_otsu if wanting to add in a morphological operation
        otsu_th, fly_mask= cv2.threshold(img_fly,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)

        # (Optional) -> Morphology: erosion
        strel = np.ones((5,5),np.uint8)  

        # maskFly = cv2.dilate(img_otsu, strel)
        # perform an erosing
        ## MORPHOLOCICAL OPERATION IN ORDER TO INCORPORATE SMOOTHING!
        # fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)


        # get the contours
        contours,hierarchy = cv2.findContours(fly_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        # utilize find_fly_angle function to get max area and contour
        max_contour, max_area = get_max_area_contour(contours)
        # determine the centroid using find_fly_angle function
        Cx, Cy = get_centroid(cv2.moments(max_contour))# centroid x,y coordinates
        ## centroid
        centroid = (int(Cx), int(Cy))
        # determine the angle and body vectory using finde_fly_angle function
        angle, body_vect = get_angle_and_body_vector(cv2.moments(max_contour))
        # Get bounding box and find diagonal - used for drawing body axis
        bbox = cv2.boundingRect(max_contour)
        bbox_diag = np.sqrt(bbox[2]**2 + bbox[3]**2)
        # Create points for drawing axis fly in contours image 
        axis_length = 0.75*bbox_diag
        body_axis_pt_0 = int(Cx+ axis_length*body_vect[0]), int(Cy+ axis_length*body_vect[1])
        body_axis_pt_1 = int(Cx- axis_length*body_vect[0]), int(Cy - axis_length*body_vect[1])
        print(body_axis_pt_0)

        # Compute cirlce mask
        mask_radius = int(.95*height/2.0)
        print('mask radius',mask_radius)
        vals_x = np.arange(0.0,width)
        vals_y = np.arange(0.0,height)
        grid_x, grid_y = np.meshgrid(vals_x, vals_y)    ## plot the centroid
        # plt.plot(centroid[0],centroid[1],marker = '.',color = 'r' ,markersize= 10)

        ## plot arrow of direction
        endpoint_x = centroid[0] + axis_length*body_vect[0]
        endpoint_y = centroid[1] - axis_length*body_vect[1]
        # plmalet.arrow(body_axis_pt_0[0],body_axis_pt_0[1],body_axis_pt_1[0],body_axis_pt_1[1],
        #             width=1,  # Width of the arrow
        #             head_width=30,  # Width of the arrow head
        #             head_length=35,  # Length of the arrow head
        #             fc='blue',  # Fill color of the arrow
        #             ec='blue',  # Edge color of the arrow
        #             linestyle='-',  # Style of the arrow
        #             length_includes_head=True

        #             )
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)

        # ##  plot the image
        # plt.imshow(img_fly,cmap ='gray')
        
        circ_mask = (grid_x - width/2.0 + 0.5)**2 + (grid_y - height/2.0 + 0.5)**2 < (mask_radius)**2



        # Get matrices for shifting (centering) and rotating the image
        shift_mat = np.matrix([[1.0, 0.0, (mid_x - Cx)], [0.0, 1.0, (mid_y - Cy)]]) 
        rot_mat = cv2.getRotationMatrix2D((mid_x, mid_y),np.rad2deg(angle),1.0)

        # Shift and rotate the original image
        shifted_image = cv2.warpAffine(img_fly, shift_mat, image_cvsize)
        rotated_image = cv2.warpAffine(shifted_image,rot_mat,image_cvsize)

        # Shift and rotate threshold image. 
        shifted_threshold_image = cv2.warpAffine(fly_mask, shift_mat, image_cvsize)
        rotated_threshold_image = cv2.warpAffine(shifted_threshold_image,rot_mat,image_cvsize)
        rotated_threshold_image = rotated_threshold_image*circ_mask
        rval, rotated_threshold_image  = cv2.threshold(rotated_threshold_image,0,256,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
        # Get orientation discriminant and flip image if needed 
        orient_ok, orient_discrim= is_orientation_ok(rotated_threshold_image,2)
        if not orient_ok:
            rot_180_mat = cv2.getRotationMatrix2D((mid_x, mid_y),-180.0,1.0)
            rotated_image = cv2.warpAffine(rotated_image,rot_180_mat,image_cvsize)
            rotated_threhold_image = cv2.warpAffine(rotated_threshold_image,rot_180_mat,image_cvsize)
            angle += np.deg2rad(-180.0)

        # angle = normalize_angle_range(angle)
        # print("angle final", angle)        fly_mask = cv2.morphologyEx( cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, strel), cv2.MORPH_OPEN, strel)

        # The images will be kept in grayscale
        # Correct contours
        max_cont_fix = max_contour.reshape(-1,2)
        ## plot the contour
        # plt.plot(max_cont_fix[:,0],max_cont_fix[:,1],color = 'r', linewidth = 2)

        # Here we will save the information to the corresponding dictionary key
        fly_gender = type_list[idx]
        img_dict[fly_gender]['img'] = img_fly
        img_dict[fly_gender]['centroid'] = centroid
        img_dict[fly_gender]['body_axis_pt_0']= body_axis_pt_0
        img_dict[fly_gender]['body_axis_pt_1']= body_axis_pt_1
        img_dict[fly_gender]['max contour']= max_cont_fix




    if display == True:
        # print(max_cont_fix)
        plt.figure(figsize=(24,24))
        ## will create 4 image panels
        ### 1 the original male image
        plt.subplot(2,2,1)
        plt.title(f"Original {fly1} Image", fontsize = 20)
        plt.axis('off')
        plt.imshow(img_dict['male']['img'],cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.tiangletle("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 3 original male image with the contour and line    ## plot the centroid
        plt.subplot(2,2,2)
        plt.title(f"{fly1} Fly Contour and Orientation", fontsize = 20)
        
        plt.plot(img_dict['male']['centroid'][0],img_dict['male']['centroid'][1],marker = '.',color = 'r' ,markersize= 10)
        ## plot the contour
        plt.plot(img_dict['male']['max contour'][:,0],img_dict['male']['max contour'][:,1],color = 'r', linewidth = 2)
        # cv2.line(img_fly, body_axis_pt_0, body_axis_pt_1, (0,0,255), 2)
        print(img_dict['male']['body_axis_pt_0'])
        plt.plot((img_dict['male']['body_axis_pt_0'][0],img_dict['male']['body_axis_pt_1'][0]),(img_dict['male']['body_axis_pt_0'][1],img_dict['male']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
        # cv2.circle(img_fly, (int(centroid[0] + np.cos(angle)*mask_radius),int(centroid[1] + np.sin(angle)*mask_radius)), 10, (0,255,255), -1)
        plt.axis('off')
        plt.imshow(img_dict['male']['img'],cmap = 'gray')
        
        ### 3 the original female image
        plt.subplot(2,2,3)
        plt.title(f"Original {fly2} Image", fontsize = 20)
        plt.axis('off')
        plt.imshow(img_dict['female']['img'],cmap = 'gray')
        # ### 2 optional show the image mask
        # plt.subplot(2,2,2)
        # plt.tiangletle("Fly Mask", fontsize = 10)
        # masked_filled_fly = get_fill_flymask(fly_mask, max_contour)
        # plt.imshow(masked_filled_fly,cmap ='gray')
        ### 4 original female image with the contour and line    ## plot the centroid
        plt.subplot(2,2,4)
        plt.title(f"{fly2} Fly Contour and Orientation", fontsize = 20)
        ## plot the contour
        plt.plot(img_dict['female']['max contour'][:,0],img_dict['female']['max contour'][:,1],color = 'r', linewidth = 2)
        ## plot the centerline
        plt.plot((img_dict['female']['body_axis_pt_0'][0],img_dict['female']['body_axis_pt_1'][0]),(img_dict['female']['body_axis_pt_0'][1],img_dict['female']['body_axis_pt_1'][1]), color = 'r', linewidth =2)
        ## get rid of the axis
        plt.axis('off')
        ##  plot the image
        plt.imshow(img_fly,cmap ='gray')
        plt.savefig('figs/2by2_otsu.svg',facecolor = 'white')


