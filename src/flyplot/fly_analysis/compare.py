from flyplot.plotting.flyviz import contour_hrz_matrix
from flyplot.plotting import polarplot as polarplt
import cv2
import numpy as np
import copy
import matplotlib.pyplot as plt

def img_label(filename):
    """
    For each image we will need to ensure that the proper rotation is performed such that it is in the 
    top down rotation
    
    """
    fly_mask_orig, fly_mask_hrz,max_contour_hrz,centroid_hrz, body_axis_pt_0_hrz,body_axis_pt_1_hrz,angle_og,angle_rot= contour_hrz_matrix(filename)
    # mask_fly_norot = get_objectmask(fly_mask_hrz,max_contour_hrz)
    mask_fly_rot = cv2.rotate(copy.deepcopy(fly_mask_hrz),cv2.ROTATE_180)
    dict_img ={
        "n": fly_mask_hrz,
        "r":mask_fly_rot
    }
    #     for i in range(1):
    ## test various rotations (0 or 180)
    plt.figure(figsize=(20,20))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(fly_mask_orig,cmap='gray')
    print(f"<< Original Angle: {polarplt.deg360to180(np.rad2deg(angle_og))} >>")
    plt.subplot(1,3,2)
    plt.title("No Rotation")
    # print("No Rotation Angle:",np.rad2deg(angle))
    print(f"<< No Rotation Angle: {polarplt.deg360to180(np.rad2deg(angle_og))} >>")
    plt.imshow(fly_mask_hrz,cmap='gray')
    plt.subplot(1,3,3)
    plt.title("Rotation")
    print(f"<< Rotation Angle: {polarplt.deg360to180(np.rad2deg(angle_og)-180)} >>")
    plt.imshow(mask_fly_rot,cmap='gray')
    plt.show()
    
    fly_up = input("Input the name of No Rotation or Rotation depending on which one display fly pointed up. ")
    print("INPUT:",fly_up)
    mask_fly = dict_img[fly_up]
    
    return mask_fly

## Comparison between two images
def mse(img1, img2):
    area_px = img1.shape[0]*img1.shape[1]
    px_diff = cv2.subtract(img1, img2)
    sum_err = np.sum(px_diff**2)
    mse = sum_err/(float(area_px))
    return mse
def data_mse(data,avg_mask):
    data_results = {
        "file":[],
        "type":[],
        "MSE":[],
        "Original Angle":[],
        "Adjusted Angle":[]
    }
    for file in data:
        name = file.split('/')[-1].strip('.png')
        mask_fly= contour_hrz_matrix(file)[1].astype(float)
        angle_og = contour_hrz_matrix(file)[6]
        ## No Rotation or Rotation
        d = {
            "MSE No Rotation":mse(avg_mask,mask_fly),
            "MSE Rotation": mse(avg_mask,cv2.rotate(mask_fly,cv2.ROTATE_180))
        }
        # print(d)
        
        ## Save the min MSE Value
        data_results["file"].append(name)
        img_rot_type = list(d.keys())[list(d.values()).index(min(list(d.values())))]
        if img_rot_type == "MSE No Rotation":
            angle_og_deg= polarplt.deg360to180(np.rad2deg(angle_og))
            angle_adj = polarplt.deg360to180(np.rad2deg(angle_og))
        else:
            angle_og_deg = polarplt.deg360to180(np.rad2deg(angle_og))
            angle_adj = polarplt.deg360to180(np.rad2deg(angle_og)-180)

        data_results['type'].append(img_rot_type)
        data_results["MSE"].append((d["MSE No Rotation"],d["MSE Rotation"])) # both MSE values...
        data_results['Original Angle'].append(angle_og_deg)
        data_results['Adjusted Angle'].append(angle_adj)
        print("----------------")
    return data_results