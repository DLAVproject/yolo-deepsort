from __future__ import division

import tensorflow
print(tensorflow.__version__)

import argparse, time, logging, os, math, tqdm, cv2

import math
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints, plot_image

from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose

import heapq

def pose_normalization(x):
    def retrain_only_body_joints(x_input):
        x0 = x_input.copy()
        #x0 = x0[2:2+13*2]
        x0 = x0[2:2+13*2] # disregards face-related points
        return x0

    def normalize(x_input):
        # Separate original data into x_list and y_list
        lx = []
        ly = []
        N = len(x_input)
        i = 0
        while i<N:
            lx.append(x_input[i])
            ly.append(x_input[i+1])
            i+=2
        lx = np.array(lx)
        ly = np.array(ly)

        # Get rid of undetected data (=0)
        non_zero_x = []
        non_zero_y = []
        for i in range(int(N/2)):
            if lx[i] != 0:
                non_zero_x.append(lx[i])
            if ly[i] != 0:
                non_zero_y.append(ly[i])
        if len(non_zero_x) == 0 or len(non_zero_y) == 0:
            return np.array([0] * N)

        # Normalization x/y data according to the bounding box
        origin_x = np.min(non_zero_x)
        origin_y = np.min(non_zero_y)
        len_x = np.max(non_zero_x) - np.min(non_zero_x)
        len_y = np.max(non_zero_y) - np.min(non_zero_y)
        x_new = []
        for i in range(int(N/2)):
            if (lx[i] + ly[i]) == 0:
                x_new.append(-1)
                x_new.append(-1)
            else:
                x_new.append((lx[i] - origin_x) / len_x)
                x_new.append((ly[i] - origin_y) / len_y)
        return x_new

    x_body_joints_xy = retrain_only_body_joints(x)
    x_body_joints_xy = normalize(x_body_joints_xy)
    return x_body_joints_xy

def drawActionResult(img_display, skeleton, str_action_type):
    font = cv2.FONT_HERSHEY_SIMPLEX 

    minx = 999
    miny = 999
    maxx = -999
    maxy = -999
    i = 0
    NaN = 0

    while i < len(skeleton):
        if not(skeleton[i]==NaN or skeleton[i+1]==NaN):
            minx = min(minx, skeleton[i])
            maxx = max(maxx, skeleton[i])
            miny = min(miny, skeleton[i+1])
            maxy = max(maxy, skeleton[i+1])
        i+=2

    minx = int(minx * img_display.shape[1])
    miny = int(miny * img_display.shape[0])
    maxx = int(maxx * img_display.shape[1])
    maxy = int(maxy * img_display.shape[0])
    print(minx, miny, maxx, maxy)
    
    # Draw bounding box
    # drawBoxToImage(img_display, [minx, miny], [maxx, maxy])
    img_display = cv2.rectangle(img_display,(minx, miny),(maxx, maxy),(0,255,0), 4)

    # Draw text at left corner


    box_scale = max(0.5, min(2.0, (1.0*(maxx - minx)/img_display.shape[1] / (0.3))**(0.5) ))
    fontsize = 1.5 * box_scale
    linewidth = int(math.ceil(3 * box_scale))

    TEST_COL = int( minx + 5 * box_scale)
    TEST_ROW = int( miny - 10 * box_scale)

    img_display = cv2.putText(img_display, str_action_type, (TEST_COL, TEST_ROW), font, fontsize, (0, 0, 255), linewidth, cv2.LINE_AA)

    return img_display

def match_pose_output_to_classifier_input(pred_coords):

    alphapose_resnet_joint_odering_dict = {"nose":0, 
                                        "left_eye":0,
                                        "right_eye":0,
                                        "left_ear":0,
                                        "right_ear":0,
                                        "left_shoulder":0,
                                        "right_shoulder":0,
                                        "left_elbow":0,
                                        "right_elbow":0,
                                        "left_wrist":0,
                                        "right_wrist":0,
                                        "left_hip":0,
                                        "right_hip":0,
                                        "left_knee":0,
                                        "right_knee":0,
                                        "left_ankle":0,
                                        "right_ankle":0}  

    tf_pose_est_joint_odering_dict =      {"nose":0, 
                                        "neck":0,
                                        "right_shoulder":0,
                                        "right_elbow":0,
                                        "right_wrist":0,
                                        "left_shoulder":0,
                                        "left_elbow":0,
                                        "left_wrist":0,
                                        "right_hip":0,
                                        "right_knee":0,
                                        "right_ankle":0,
                                        "left_hip":0,
                                        "left_knee":0,
                                        "left_ankle":0,
                                        "right_eye":0,
                                        "left_eye":0,
                                        "right_ear":0,
                                        "left_ear":0}  

    for i,key in enumerate(alphapose_resnet_joint_odering_dict):
        alphapose_resnet_joint_odering_dict[key] = pred_coords[i].asnumpy()
    alphapose_resnet_joint_odering_dict["neck"] = 0.5*(alphapose_resnet_joint_odering_dict["left_shoulder"] + alphapose_resnet_joint_odering_dict["right_shoulder"])
    for key in tf_pose_est_joint_odering_dict.keys():
        tf_pose_est_joint_odering_dict[key] = alphapose_resnet_joint_odering_dict[key]

    skeleton_classifier_input = []

    for value in tf_pose_est_joint_odering_dict.values():
        skeleton_classifier_input.append(value[0])
        skeleton_classifier_input.append(value[1])
        
    skeleton_classifier_input = np.array(skeleton_classifier_input)

    return skeleton_classifier_input

class ActionClassifier(object):
    
    def __init__(self, model_path):
        from keras.models import load_model

        self.dnn_model = load_model(model_path)
        self.action_dict = ["kick", "punch", "squat", "stand", "wave"]
        #self.action_dict = ["stand", "wave"]

    def predict(self, skeleton):

        # Preprocess data
        tmp = pose_normalization(skeleton)
        skeleton_input = np.array(tmp).reshape(-1, len(tmp))
            
        # Predicted label: int & string
        predicted_idx = np.argmax(self.dnn_model.predict(skeleton_input))
        predicted_label = self.action_dict[predicted_idx]

        return predicted_label

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.

    '''

    # Get the required landmarks coordinates.
    x1, y1 = landmark1
    x2, y2 = landmark2
    x3, y3 = landmark3

    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    angle = np.abs(angle)
    # Check if the angle is less than zero.
    if angle > 180.0:

        angle = 360-angle
    
    # Return the calculated angle.
    return angle

def classifyPose(kp_array, output_image=None, display=False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        kp_array: A list of detected landmarks of the person whose pose needs to be classified.
        output_image: A image of the person with the detected pose landmarks drawn.
        display: A boolean value that is if set to true the function displays the resultant image with the pose label 
        written on it and returns nothing.
    Returns:
        output_image: The image with the detected pose landmarks drawn and pose label written.
        label: The classified pose label of the person in the output_image.

    '''
    
    # Initialize the label of the pose. It is not known at this stage.
    label = 'Unknown Pose'

    # Specify the color (Red) with which the label will be written on the image.
    color = (0, 0, 255)
    
    # Calculate the required angles.
    #----------------------------------------------------------------------------------------------------------------
    
    # Get the angle between the left shoulder, elbow and wrist points. 
    left_elbow_angle = calculateAngle(kp_array[5],
                                      kp_array[7],
                                      kp_array[9])
    
    # Get the angle between the right shoulder, elbow and wrist points.
    right_elbow_angle = calculateAngle(kp_array[6],
                                       kp_array[8],
                                       kp_array[10])
    
    # Get the angle between the left elbow, shoulder and hip points.
    left_shoulder_angle = calculateAngle(kp_array[7],
                                         kp_array[5],
                                         kp_array[11])
    # Get the angle between the right hip, shoulder and elbow points.
    right_shoulder_angle = calculateAngle(kp_array[12],
                                          kp_array[6],
                                          kp_array[8])
    # Check if the both arms are straight.
    if left_elbow_angle > 125 and left_elbow_angle < 220 and right_elbow_angle > 125 and right_elbow_angle < 220:
        #label = 'T Pose'
        # Check if shoulders are at the required angle.
        if left_shoulder_angle > 70 and left_shoulder_angle < 110 and right_shoulder_angle > 70 and right_shoulder_angle < 110:
            label = 'T Pose'

    if right_elbow_angle > 50 and right_elbow_angle < 130 and right_shoulder_angle > 70 and right_shoulder_angle < 110:
        label = 'Power to the People'                  
    
    # Check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        # Update the color (to green) with which the label will be written on the image.
        color = (0, 255, 0)  
    
    # Write the label on the output image. 
    #cv2.putText(output_image, label, (10, 30),cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
    
    # Check if the resultant image is specified to be displayed.
    if display:
    
        # Display the resultant image.
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
    else:
        
        # Return the output image and the classified label.
        return label

