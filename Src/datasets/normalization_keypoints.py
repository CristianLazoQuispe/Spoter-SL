import gc
import ast
import tqdm
import time
import glob
import json
import torch
import pandas as pd
import numpy as np
from collections import Counter
import torch.utils.data as torch_data
from torch.utils.data import Dataset
import logging
import random
import copy

from .augmentations import augmentations

import cv2

import h5py
def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf

####################################################################
# Function that helps to see keypoints in an image
####################################################################
def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    cv2.imwrite(f'Results/images/keypoints/keypoint_{tag}.jpg', img)

##########################################################
# Process used to normalize the pose
##########################################################
def normalize_pose(data, body_dict):

    sequence_size = data.shape[0]
    valid_sequence = True

    last_starting_point, last_ending_point = None, None

    for sequence_index in range(sequence_size):

        # Prevent from even starting the analysis if some necessary elements are not present
        if (data[sequence_index][body_dict['pose_left_shoulder']][0] == 0.0 or data[sequence_index][body_dict['pose_right_shoulder']][0] == 0.0):
            if not last_starting_point:
                valid_sequence = False
                continue

            else:
                starting_point, ending_point = last_starting_point, last_ending_point
    
        else:

            # NOTE:
            #
            # While in the paper, it is written that the head metric is calculated by halving the shoulder distance,
            # this is meant for the distance between the very ends of one's shoulder, as literature studying body
            # metrics and ratios generally states. The Vision Pose Estimation API, however, seems to be predicting
            # rather the center of one's shoulder. Based on our experiments and manual reviews of the data, employing
            # this as just the plain shoulder distance seems to be more corresponding to the desired metric.
            #
            # Please, review this if using other third-party pose estimation libraries.

            if data[sequence_index][body_dict['pose_left_shoulder']][0] != 0 and data[sequence_index][body_dict['pose_right_shoulder']][0] != 0:
                
                left_shoulder = data[sequence_index][body_dict['pose_left_shoulder']]
                right_shoulder = data[sequence_index][body_dict['pose_right_shoulder']]

                shoulder_distance = ((((left_shoulder[0] - right_shoulder[0]) ** 2) + (
                                       (left_shoulder[1] - right_shoulder[1]) ** 2)) ** 0.5)

                mid_distance = (0.5,0.5)#(left_shoulder - right_shoulder)/2
                head_metric = shoulder_distance/2
            '''
            # use it if you have the neck keypoint
            else:
                neck = (data["neck_X"][sequence_index], data["neck_Y"][sequence_index])
                nose = (data["nose_X"][sequence_index], data["nose_Y"][sequence_index])
                neck_nose_distance = ((((neck[0] - nose[0]) ** 2) + ((neck[1] - nose[1]) ** 2)) ** 0.5)
                head_metric = neck_nose_distance
            '''
            # Set the starting and ending point of the normalization bounding box
            starting_point = [mid_distance[0] - 3 * head_metric, data[sequence_index][body_dict['pose_right_eye']][1] - (head_metric / 2)]
            ending_point = [mid_distance[0] + 3 * head_metric, mid_distance[1] + 4.5 * head_metric]

            last_starting_point, last_ending_point = starting_point, ending_point

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):
            
            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - ending_point[1]) / (starting_point[1] -
                                                                                    ending_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = 1 - normalized_y
            
    return data
################################################
# Function that normalize the hands (but also the face)
################################################
def normalize_hand(data, body_section_dict):
    """
    Normalizes the skeletal data for a given sequence of frames with signer's hand pose data. The normalization follows
    the definition from our paper.
    :param data: Dictionary containing key-value pairs with joint identifiers and corresponding lists (sequences) of
                that particular joints coordinates
    :return: Dictionary with normalized skeletal data (following the same schema as input data)
    """

    sequence_size = data.shape[0]
    
    # Treat each element of the sequence (analyzed frame) individually
    for sequence_index in range(sequence_size):

        # Retrieve all of the X and Y values of the current frame
        landmarks_x_values = data[sequence_index][:, 0]
        landmarks_y_values = data[sequence_index][:, 1]

        # Prevent from even starting the analysis if some necessary elements are not present
        #if not landmarks_x_values or not landmarks_y_values:
        #    continue

        # Calculate the deltas
        width, height = max(landmarks_x_values) - min(landmarks_x_values), max(landmarks_y_values) - min(
            landmarks_y_values)
        if width > height:
            delta_x = 0.1 * width
            delta_y = delta_x + ((width - height) / 2)
        else:
            delta_y = 0.1 * height
            delta_x = delta_y + ((height - width) / 2)

        # Set the starting and ending point of the normalization bounding box
        starting_point = (min(landmarks_x_values) - delta_x, min(landmarks_y_values) - delta_y)
        ending_point = (max(landmarks_x_values) + delta_x, max(landmarks_y_values) + delta_y)

        # Normalize individual landmarks and save the results
        for pos, kp in enumerate(data[sequence_index]):

            # Prevent from trying to normalize incorrectly captured points
            if data[sequence_index][pos][0] == 0 or (ending_point[0] - starting_point[0]) == 0 or (
                    starting_point[1] - ending_point[1]) == 0:
                continue

            normalized_x = (data[sequence_index][pos][0] - starting_point[0]) / (ending_point[0] -
                                                                                    starting_point[0])
            normalized_y = (data[sequence_index][pos][1] - starting_point[1]) / (ending_point[1] -
                                                                                    starting_point[1])

            data[sequence_index][pos][0] = normalized_x
            data[sequence_index][pos][1] = normalized_y

    return data

###################################################################################
# This function normalize the body and the hands separately
# body_section has the general body part name (ex: pose, face, leftHand, rightHand)
# body_part has the specific body part name (ex: pose_left_shoulder, face_right_mouth_down, etc)
###################################################################################
def normalize_pose_hands_function(data, body_section, body_part):
    data = copy.deepcopy(data)
    pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
    face = [pos for pos, body in enumerate(body_section) if body == 'face']
    leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
    rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']

    body_section_dict = {body:pos for pos, body in enumerate(body_part)}

    assert len(pose) > 0 and len(leftHand) > 0 and len(rightHand) > 0 #and len(face) > 0

    prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"before")

    for index_video in range(len(data)):
        data[index_video][:,pose,:] = normalize_pose(data[index_video][:,pose,:], body_section_dict)
        #data[index_video][:,face,:] = normalize_hand(data[index_video][:,face,:], body_section_dict)
        data[index_video][:,leftHand,:] = normalize_hand(data[index_video][:,leftHand,:], body_section_dict)
        data[index_video][:,rightHand,:] = normalize_hand(data[index_video][:,rightHand,:], body_section_dict)

    prepare_keypoints_image(data[2][0][leftHand+rightHand+pose,:],"after")

    kp_bp_index = {'pose':pose,
                   'left_hand':leftHand,
                   'rigth_hand':rightHand}

    return data, kp_bp_index, body_section_dict



