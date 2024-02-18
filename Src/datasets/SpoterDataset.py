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

from .augmentations import augmentations_torch
#from .Lsp_dataset import *
from . import normalization_keypoints as normalization

import cv2

import h5py
def get_data_from_h5(path):
    hf = h5py.File(path, 'r')
    return hf


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
        video_dataset.append(data_video)
        labels_dataset.append(data_label)
        video_name_dataset.append(data_video_name.encode('utf-8'))

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

    return video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, df_keypoints['Section'], section_keypoints,df_keypoints

class SpoterDataset(Dataset):
    """Advanced object representation of the HPOES dataset for loading hand joints landmarks utilizing the Torch's
    built-in Dataset properties"""

    data: [np.ndarray]  # type: ignore
    labels: [np.ndarray]  # type: ignore

    def __init__(self, dataset_filename: str,keypoints_model:str,  transform=None, has_augmentation=False,
                 augmentations_prob=0.5,landmarks_ref= 'Data/Mapeo landmarks librerias.csv',
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

        video_dataset, video_name_dataset, labels_dataset, encoded_dataset, dict_labels_dataset, inv_dict_labels_dataset, body_section, body_part,df_keypoints = get_dataset_from_hdf5(path=dataset_filename,
                                keypoints_model=keypoints_model,
                                landmarks_ref=landmarks_ref,
                                keypoints_number = keypoints_number,
                                threshold_frecuency_labels =0,
                                list_labels_banned =self.list_labels_banned,
                                dict_labels_dataset=dict_labels_dataset,
                                inv_dict_labels_dataset=inv_dict_labels_dataset)
        # HAND AND POSE NORMALIZATION
        print("# HAND AND POSE NORMALIZATION")
        print("# HAND AND POSE NORMALIZATION")
        print("# HAND AND POSE NORMALIZATION")

        pose = [pos for pos, body in enumerate(body_section) if body == 'pose' or body == 'face']
        face = [pos for pos, body in enumerate(body_section) if body == 'face']
        leftHand = [pos for pos, body in enumerate(body_section) if body == 'leftHand']
        rightHand = [pos for pos, body in enumerate(body_section) if body == 'rightHand']


        normalization.prepare_keypoints_image(video_dataset[2][0][leftHand+rightHand+pose,:],"befor2")
        video_dataset_normalized, keypoint_body_part_index, body_section_dict = normalization.normalize_pose_hands_function(video_dataset, body_section, body_part)
        normalization.prepare_keypoints_image(video_dataset[2][0][leftHand+rightHand+pose,:],"after2")

        self.df_keypoints = df_keypoints
        self.keypoint_body_part_index = keypoint_body_part_index
        self.body_section_dict = body_section_dict

        self.data = video_dataset
        self.data_normalized = video_dataset_normalized
        self.video_name = video_name_dataset
        self.labels = encoded_dataset
        self.label_freq = Counter(self.labels)
        
        
        self.text_labels = list(labels_dataset)
        self.transform = transform
        self.dict_labels_dataset = dict_labels_dataset
        self.inv_dict_labels_dataset = inv_dict_labels_dataset

        self.factor = factor
        max_frequency = self.factor*max(self.label_freq.values())
        # Calcular los factores de ajuste
        self.factors = {label: int(max_frequency / count+0.5) for label, count in self.label_freq.items()}

        self.map_ids_augmentation = self.get_ids_augmentation()

        print(keypoint_body_part_index, body_section_dict)
        self.augmentation = augmentations_torch.augmentation(keypoint_body_part_index, body_section_dict,device='cuda')
        self.augmentations_prob = augmentations_prob
        self.has_augmentation = has_augmentation
        
    def get_ids_augmentation(self):
        map_ids_augmentation = {}
        cnt = 0
        for idx in range(len(self.labels)):
            label = self.labels[idx]
            n_factor = self.factors[label]
            for idy in range(n_factor):
                map_ids_augmentation[cnt] = idx
                cnt+=1
        return map_ids_augmentation        

    def __getitem__(self, idx):
        """
        Allocates, potentially transforms and returns the item at the desired index.

        :param idx: Index of the item
        :return: Tuple containing both the depth map and the label
        """
        return idx

    def __len__(self):
        if self.has_augmentation:
            return len(self.map_ids_augmentation)
        else:
            return len(self.labels)

