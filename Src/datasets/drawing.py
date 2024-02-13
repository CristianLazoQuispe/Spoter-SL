import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import copy

class drawing:

    def __init__(self,w = 256,h = 256,path_points= '',radius=1):
        """
        # THIS IS A SAMPLE TO USE THESE FUNCTIONS
        
        # Example Usage:
        #
        # connections = np.moveaxis(np.array(get_edges_index('54')), 0, 1)
        # kp_frame = data[instance_pos][frame_pos]
        # prepare_keypoints_image(kp_frame, connections, -1, "pose")
        
        # ADDITIONAL COMMENTS
        #
        # Locate the "points_54.csv" file in the same folder as this script.
        #
        # These functions are used to draw the keypoints of just one frame.
        # If the data doesn't work, try using np.transpose(data, (0,2,1)) in the dataset.
        # Then get one frame with kp_frame = data[instance_pos][frame_pos].
        """
        self.w = w
        self.h = h
        self.path_points = path_points #./points_54.csv
        self.radius = radius
        self.img_base = np.zeros((w,h, 3), np.uint8)

        if self.path_points != "":
            self.connections = self.get_connections()
    
    # Show just points
    def draw_points(self,keypoints):
        
        # DRAW POINTS
        img = np.zeros((self.w, self.h, 3), np.uint8)
    
        for n, coords in enumerate(keypoints):
    
            cor_x = int(coords[0] * self.w)
            cor_y = int(coords[1] * sef.h)
    
            cv2.circle(img, (cor_x, cor_y), self.radius, (0, 0, 255), -1)
    
        return img
    
    # Shows points with connections
    def draw_lines(self,keypoints, text_left='', text_right=''):
        # This variable is used to draw points conections
        
        part_line = {}
    
        # DRAW POINTS
        img= np.zeros((self.w, self.h, 3), np.uint8)
    
        # To print numbers
        fontScale = 0.5
        color = (0, 255, 0)
        thickness = 1
    
        # To print the text
        if text_right!="":
            img = cv2.putText(img, str(pos_rel), (220, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale, color, thickness, cv2.LINE_AA)
        if text_left!="":
            img = cv2.putText(img, addText, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale, color, thickness, cv2.LINE_AA)

        n = len(keypoints)
        
        # DRAW JOINT LINES
        for start_p, end_p in self.connections:
            if start_p <n and end_p <n:
                cor_x = int(keypoints[start_p][0] * self.w)
                cor_y = int(keypoints[start_p][1] * self.h)
                start_p = (cor_x,cor_y)
                
                cor_x = int(keypoints[end_p][0] * self.w)
                cor_y = int(keypoints[end_p][1] * self.h)
                end_p = (cor_x,cor_y)

                cv2.line(img, start_p, end_p, (255,100,100), 2)
        for coords in keypoints:
    
            cor_x = int(coords[0] * self.w)
            cor_y = int(coords[1] * self.h)
            cv2.circle(img, (cor_x,cor_y), self.radius, (255, 255, 255), -1)
    
        return img


    def get_connections(self):
        
        points_joints_info = pd.read_csv(self.path_points)
        # we subtract one because the list is one start (we wanted it to start in zero)
        ori = points_joints_info.origin-1
        tar = points_joints_info.tarjet-1
    
        ori = np.array(ori)
        tar = np.array(tar)
    
        connections = np.array([ori,tar])
        connections = np.moveaxis(np.array(connections), 0, 1)
    
        return connections
