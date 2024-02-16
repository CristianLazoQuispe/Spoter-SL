import matplotlib.animation as animation
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import imageio
import pandas as pd
import numpy as np
import cv2
import copy
import os

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
            img = cv2.putText(img, text_right, (200, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale, color, thickness, cv2.LINE_AA)
        if text_left!="":
            img = cv2.putText(img, text_left, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 
                              fontScale, color, thickness, cv2.LINE_AA)

        n = len(keypoints)
        #print("keypoints.shape",keypoints.shape)
        #print("keypoints.shape",n)
        # DRAW JOINT LINES
        for start_p, end_p in self.connections:
            if start_p <n and end_p <n:
                #print("start_p",start_p)
                #print("keypoints",keypoints[start_p] ,self.w)
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

    def save_video(self,list_images,suffix='train'):
        height, width, _ = list_images[0].shape
        
        # Definir el nombre del archivo de video de salida
        video_output = f'Results/images/keypoints/matrix_25_gloss_{suffix}.mp4'
        
        # Configurar el objeto VideoWriter de OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_output, fourcc, 1, (width, height))
        
        # Escribir cada imagen en el video
        for image in list_images:
            # Asegúrate de que la imagen sea del tipo uint8
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            image = image.astype('uint8')
            # Escribir la imagen en el video
            video_writer.write(image)
        
        # Liberar el objeto VideoWriter
        video_writer.release()
        
        print("Video creado con éxito:", video_output)
        return video_output

    def get_video_frames_25_glosses(self,list_depth_map_original,list_label_name_original,suffix='train',save_gif = True):
        
        list_depth_map = []
        list_label_name = []
        max_frames = 0
        
        for depth_map,video_name in zip(list_depth_map_original,list_label_name_original):
            depth_map = depth_map.cpu().numpy()
            depth_map  = depth_map[0]
            label_name       = video_name[0].split("/")[-1].split(".")[0]
            #print(f"depth_map  : {depth_map.shape}")
            max_frames = max(max_frames,depth_map.shape[0])
            list_depth_map.append(depth_map)
            list_label_name.append(label_name)
        
        list_images = []
        
        for id_frame in tqdm(range(max_frames)):
            fig, axs = plt.subplots(5, 5, figsize=(20, 20))
            for i in range(5):
                for j in range(5):
                    depth_map = list_depth_map[i*5 + j] 
                    label_name = list_label_name[i*5 + j] 
        
                    if id_frame<depth_map.shape[0]:
                        kp_frame = depth_map[id_frame]+0.5
                        img = self.draw_lines(kp_frame,text_left=label_name,text_right=str(id_frame))
                    else:
                        kp_frame = depth_map[-1]+0.5
                        id_frame_new = depth_map.shape[0]-1
                        img = self.draw_lines(kp_frame,text_left=label_name,text_right=str(id_frame_new)+"_last")
                        #img= np.zeros((256,256, 3), np.uint8)
                        
                    axs[i, j].imshow(img)
                    axs[i, j].axis('off')
            plt.tight_layout() 
            plt.savefig(f'Results/images/keypoints/matrix_25_gloss_{suffix}_{id_frame}.jpg')
            plt.close(fig)
            img = plt.imread(f'Results/images/keypoints/matrix_25_gloss_{suffix}_{id_frame}.jpg')    
            os.remove(f'Results/images/keypoints/matrix_25_gloss_{suffix}_{id_frame}.jpg')
            list_images.append(img) 
        filename = ''
        if save_gif:
            
            # Crear el GIF   
            print("Saving",f'Results/images/keypoints/matrix_25_gloss_{suffix}.gif')
            print("Saving",f'Results/images/keypoints/matrix_25_gloss_{suffix}.npy')
            imageio.mimsave(f'Results/images/keypoints/matrix_25_gloss_{suffix}.gif', list_images, fps=1)
            np.save(f'Results/images/keypoints/matrix_25_gloss_{suffix}.npy', list_images)

            filename = self.save_video(list_images,suffix=suffix)
        return list_images, f'Results/images/keypoints/matrix_25_gloss_{suffix}.gif'