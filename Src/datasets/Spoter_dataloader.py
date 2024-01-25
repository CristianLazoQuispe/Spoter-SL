import gc
import ast
import tqdm
import time
import h5py
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

from .augmentations import augmentations
from .Lsp_dataset import *

import cv2

def get_dataset_from_hdf5(path,keypoints_model,landmarks_ref,keypoints_number,threshold_frecuency_labels=10,list_labels_banned=[],dict_labels_dataset=None,
                         inv_dict_labels_dataset=None):

    json_file_path = "Data/meaning.json"

    print('path                       :',path)
    print('keypoints_model            :',keypoints_model)
    print('landmarks_ref              :',landmarks_ref)
    print('threshold_frecuency_labels :',threshold_frecuency_labels)
    print('list_labels_banned         :',list_labels_banned)
    
    # Prepare the data to process the dataset

    index_array_column = None #'mp_indexInArray', 'wp_indexInArray','op_indexInArray'

    print('Use keypoint model : ',keypoints_model) 
    if keypoints_model == 'openpose':
        index_array_column  = 'op_indexInArray'
    if keypoints_model == 'mediapipe':
        index_array_column  = 'mp_indexInArray'
    if keypoints_model == 'wholepose':
        index_array_column  = 'wp_indexInArray'
    print('use column for index keypoint :',index_array_column)

    assert not index_array_column is None

    # all the data from landmarks_ref
    df_keypoints = pd.read_csv(landmarks_ref, skiprows=1)

    # 29, 54 or 71 points
    if keypoints_number == 29:
        df_keypoints = df_keypoints[(df_keypoints['Selected 29']=='x' )& (df_keypoints['Key']!='wrist')]
    elif keypoints_number == 71:
        df_keypoints = df_keypoints[(df_keypoints['Selected 71']=='x' )& (df_keypoints['Key']!='wrist')]
    else:
        df_keypoints = df_keypoints[(df_keypoints['Selected 54']=='x')]

    print(" using keypoints_number: "+str(keypoints_number))
    logging.info(" using keypoints_number: "+str(keypoints_number))

    idx_keypoints = sorted(df_keypoints[index_array_column].astype(int).values)
    name_keypoints = df_keypoints['Key'].values
    section_keypoints = (df_keypoints['Section']+'_'+df_keypoints['Key']).values

    print('section_keypoints : ',len(section_keypoints),' -- uniques: ',len(set(section_keypoints)))
    print('name_keypoints    : ',len(name_keypoints),' -- uniques: ',len(set(name_keypoints)))
    print('idx_keypoints     : ',len(idx_keypoints),' -- uniques: ',len(set(idx_keypoints)))
    print('')
    print('section_keypoints used:')
    print(section_keypoints)

    # process the dataset (start)

    print('Reading dataset .. ')
    data = get_data_from_h5(path)
    #torch.Size([5, 71, 2])

    print('Total size dataset : ',len(data.keys()))
    print('Keys in dataset:', data.keys())
    video_dataset  = []
    labels_dataset = []

    video_name_dataset = []
    false_seq_dataset = []
    percentage_dataset = []
    max_consec_dataset = []

    for index in tqdm.tqdm(list(data.keys())):
        data_video = np.array(data[index]['data'])
        data_label = np.array(data[index]['label']).item().decode('utf-8')
        # F x C x K  (frames, coords, keypoitns)
        n_frames, n_axis, n_keypoints = data_video.shape

        data_video = np.transpose(data_video, (0,2,1)) #transpose to n_frames, n_keypoints, n_axis 
        if index=='0':
            print('original size video : ',data_video.shape,'-- label : ',data_label)
            print('filtering by keypoints idx .. ')
        data_video = data_video[:,idx_keypoints,:]

        if index=='0':
            print('filtered size video : ',data_video.shape,'-- label : ',data_label)

        data_video_name = np.array(data[index]['video_name']).item().decode('utf-8')
        #data_false_seq = np.array(data[index]['false_seq'])
        #data_percentage_groups = np.array(data[index]['percentage_group'])
        #data_max_consec = np.array(data[index]['max_percentage'])



        video_dataset.append(data_video)
        labels_dataset.append(data_label)
        video_name_dataset.append(data_video_name.encode('utf-8'))
        #false_seq_dataset.append(data_false_seq)
        #percentage_dataset.append(data_percentage_groups)
        #max_consec_dataset.append(data_max_consec)
        # # Get additional video attributes
        # videoname = np.array(data[index]['video_name']).item().decode('utf-8')
        # false_seq = np.array(data[index]['false_seq']).item()
        # percentage_groups = np.array(data[index]['percentage_group']).item()
        # max_consec = np.array(data[index]['max_percentage']).item()
    #     print("videoname:",videoname,"type:",type(videoname))                
    #     print("false_seq:",false_seq,"type:",type(false_seq))
    #     print("percentage_groups:",percentage_groups,"type:",type(percentage_groups))
    #     print("max_consec:",max_consec,"type:",type(max_consec))

    #     video_info.append((
    #         videoname,
    #         false_seq,
    #         percentage_groups,
    #         max_consec
    #     ))

    # print("video info shape:",len(video_info))

    del data
    gc.collect()
    
    if dict_labels_dataset is None:
        dict_labels_dataset = {}
        inv_dict_labels_dataset = {}

        for index,label in enumerate(sorted(set(labels_dataset))):
            dict_labels_dataset[label] = index
            inv_dict_labels_dataset[index] = label
    
    
    print('frecuency labels filtering ...')
    hist_labels = dict(Counter(labels_dataset))
    print('hist counter')
    print(hist_labels)

    json_data = json.dumps(inv_dict_labels_dataset, indent=4)
    with open(json_file_path, "w") as jsonfile:
        jsonfile.write(json_data)

    
    print('sorted(set(labels_dataset))  : ',sorted(set(labels_dataset)))
    print('dict_labels_dataset      :',dict_labels_dataset)
    print('inv_dict_labels_dataset  :',inv_dict_labels_dataset)
    encoded_dataset = [dict_labels_dataset[label] for label in labels_dataset]
    print('encoded_dataset:',len(encoded_dataset))

    print('label encoding completed!')

    print('total unique labels : ',len(set(labels_dataset)))
    print('Reading dataset completed!')

    return video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, df_keypoints['Section'], section_keypoints

class LSP_Dataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str,  transform=None, have_aumentation=False,has_normalization=False,
                 augmentations_prob=0.5, normalize=False,landmarks_ref= 'Data/Mapeo landmarks librerias.csv',
                dict_labels_dataset=None,inv_dict_labels_dataset=None, keypoints_number = 54,factor = 2):
        """
        Initiates the HPOESDataset with the pre-loaded data from the h5 file.

        :param dataset_filename: Path to the h5 file
        :param transform: Any data transformation to be applied (default: None)
        """
        print("*"*20)
        print("*"*20)
        print("*"*20)
        print('Use keypoint model : ',keypoints_model) 
        logging.info('Use keypoint model : '+str(keypoints_model))

        self.has_normalization = has_normalization
        self.list_labels_banned = []

        if  'AEC' in  dataset_filename:
            self.list_labels_banned += []

        if  'PUCP' in  dataset_filename:
            self.list_labels_banned += []
            self.list_labels_banned += []

        if  'WLASL' in  dataset_filename:
            self.list_labels_banned += []

        print('self.list_labels_banned',self.list_labels_banned)
        logging.info('self.list_labels_banned '+str(self.list_labels_banned))

        video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, body_section, body_part = get_dataset_from_hdf5(path=dataset_filename,
                                keypoints_model=keypoints_model,
                                landmarks_ref=landmarks_ref,
                                keypoints_number = keypoints_number,
                                threshold_frecuency_labels =0,
                                list_labels_banned =self.list_labels_banned,
                                dict_labels_dataset=dict_labels_dataset,
                                inv_dict_labels_dataset=inv_dict_labels_dataset)
        # HAND AND POSE NORMALIZATION
        video_dataset, keypoint_body_part_index, body_section_dict = normalize_pose_hands_function(video_dataset, body_section, body_part)

        self.data = video_dataset
        self.video_name = video_name_dataset
        self.labels = encoded_dataset
        self.label_freq = Counter(self.labels)
        self.factor = factor

        max_frequency = self.factor*max(self.label_freq.values())

        # Calcular los factores de ajuste
        self.factors = {label: max_frequency / count for label, count in self.label_freq.items()}
        
        self.text_labels = list(labels_dataset)
        self.transform = transform
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset
        
        self.have_aumentation = have_aumentation
        print(keypoint_body_part_index, body_section_dict)
        self.augmentation = augmentations.augmentation(keypoint_body_part_index, body_section_dict,device='cuda')
        self.augmentations_prob = augmentations_prob
        self.normalize = normalize


    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        #depth_map = torch.from_numpy(np.copy(self.data[idx]))
        #depth_map = depth_map.to('cuda')
        depth_map = torch.tensor(self.data[idx], device='cuda')#.clone()

        # Apply potential augmentations
        if self.have_aumentation and random.random() < self.augmentations_prob:

            selected_aug = random.randrange(4)

            if selected_aug == 0:
                depth_map = self.augmentation.augment_rotate(depth_map, angle_range=(-13, 13))

            if selected_aug == 1:
                depth_map = self.augmentation.augment_shear(depth_map, "perspective", squeeze_ratio=(0, 0.1))

            if selected_aug == 2:
                depth_map = self.augmentation.augment_shear(depth_map, "squeeze", squeeze_ratio=(0, 0.15))

            if selected_aug == 3:
                depth_map = self.augmentation.augment_arm_joint_rotate(depth_map, 0.3, angle_range=(-4, 4))
        video_name = self.video_name[idx].decode('utf-8')
        label = torch.Tensor([self.labels[idx]])

        if self.has_normalization:
            depth_map = depth_map - 0.5
            if self.transform:
                depth_map = self.transform(depth_map)
            
        depth_map = depth_map.to('cuda')
        label = label.to('cuda', dtype=torch.long)

        return depth_map, label, video_name

    def __len__(self):
        return len(self.labels)


