import math
import logging
import cv2
import random
import torch

import numpy as np

from torchvision import transforms



class augmentation():
    
    def __init__(self, body_type_identifiers, body_section_dict,device='gpu'):
        super().__init__()
        print(body_type_identifiers.keys())
        self.body_section_dict = body_section_dict
        self.BODY_IDENTIFIERS = body_type_identifiers['pose']
        self.HAND_IDENTIFIERS = body_type_identifiers['left_hand'] + body_type_identifiers['rigth_hand']
        
        Left_hand_id = ['pose_chest_middle_up', 'pose_left_shoulder', 'pose_left_elbow', 'pose_left_wrist']
        right_hand_id = ['pose_chest_middle_up', 'pose_right_shoulder', 'pose_right_elbow', 'pose_right_wrist']

        self.ARM_IDENTIFIERS_ORDER = [[body_section_dict[_id] for _id in Left_hand_id ],
                                      [body_section_dict[_id] for _id in right_hand_id]]

        '''
        arms_identifiers = ['pose_chest_middle_up', 'pose_right_wrist', 'pose_left_wrist','pose_right_elbow','pose_left_elbow', 'pose_left_shoulder', 'pose_right_shoulder']
        self.ARM_IDENTIFIERS_ORDER = [body_section_dict[identifiers] for identifiers in arms_identifiers]
        '''
        self.device = device

    def __random_pass(self, prob):
        return random.random() < prob


    def __numpy_to_dictionary(self, data_array: np.ndarray) -> dict:
        """
        Supplementary method converting a NumPy array of body landmark data into dictionaries. The array data must match the
        order of the BODY_IDENTIFIERS list.
        """

        output = {}

        for landmark_index, identifier in enumerate(self.BODY_IDENTIFIERS):
            output[identifier] = data_array[:, landmark_index].tolist()

        return output


    def __dictionary_to_numpy(self, landmarks_dict: dict) -> np.ndarray:
        """
        Supplementary method converting dictionaries of body landmark data into respective NumPy arrays. The resulting array
        will match the order of the BODY_IDENTIFIERS list.
        """

        output = np.empty(shape=(len(landmarks_dict["leftEar"]), len(BODY_IDENTIFIERS), 2))

        for landmark_index, identifier in enumerate(BODY_IDENTIFIERS):
            output[:, landmark_index, 0] = np.array(landmarks_dict[identifier])[:, 0]
            output[:, landmark_index, 1] = np.array(landmarks_dict[identifier])[:, 1]

        return output





    def __preprocess_row_sign(self, sign: dict) -> (dict, dict):
        """
        Supplementary method splitting the single-dictionary skeletal data into two dictionaries of body and hand landmarks
        respectively.
        """

        #sign_eval = sign

        body_landmarks = sign[:,self.BODY_IDENTIFIERS,:]
        hand_landmarks = sign[:,self.HAND_IDENTIFIERS,:]
        '''
        if "nose_X" in sign_eval:
            body_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                            for identifier in BODY_IDENTIFIERS}
            hand_landmarks = {identifier: [(x, y) for x, y in zip(sign_eval[identifier + "_X"], sign_eval[identifier + "_Y"])]
                            for identifier in HAND_IDENTIFIERS}

        else:
            body_landmarks = {identifier: sign_eval[identifier] for identifier in BODY_IDENTIFIERS}
            hand_landmarks = {identifier: sign_eval[identifier] for identifier in HAND_IDENTIFIERS}
        '''
        return body_landmarks, hand_landmarks


    def __wrap_sign_into_row(self, body_identifiers: dict, hand_identifiers: dict) -> dict:
        """
        Supplementary method for merging body and hand data into a single dictionary.
        """

        #return {**body_identifiers, **hand_identifiers}
        body_landmarks = torch.tensor(body_identifiers,device=self.device)
        hand_landmarks = torch.tensor(hand_identifiers,device=self.device)

        # Concatenar los dos tensores a lo largo de la segunda dimensión
        tensor_concatenado = torch.cat([body_landmarks, hand_landmarks], dim=1)
        return tensor_concatenado
    def __rotate(self, origin: tuple, point: tuple, angle: float):
        """
        Rotates a point counterclockwise by a given angle around a given origin.
        :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
        :param point: Landmark in the (X, Y) format to be rotated
        :param angle: Angle under which the point shall be rotated
        :return: New landmarks (coordinates)
        """

        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy
    def __rotate_torch(self, origin, tensor, angle: float):
        """
        Rotates a point counterclockwise by a given angle around a given origin.
        :param origin: Landmark in the (X, Y) format of the origin from which to count angle of rotation
        :param point: Landmark in the (X, Y) format to be rotated
        :param angle: Angle under which the point shall be rotated
        :return: New landmarks (coordinates)
        """
        if type(origin) == tuple:
            ox, oy = origin        
        else:
            # tensor of origins
            ox, oy = origin[:,0].unsqueeze(1),origin[:,1].unsqueeze(1)        

        rotated = torch.zeros_like(tensor).to(self.device)
        rotated[:,:,0] = ox + torch.cos(angle) * (tensor[:,:,0] - ox) - torch.sin(angle) * (tensor[:,:,1] - oy)
        rotated[:,:,1] = oy + torch.sin(angle) * (tensor[:,:,0] - ox) + torch.cos(angle) * (tensor[:,:,1] - oy)
        
        return rotated
        
    def augment_rotate(self, tensor, angle_range,origin = (0.5,0.5)) -> dict:
        angle = torch.tensor(math.radians(random.uniform(*angle_range)))
        rotated = self.__rotate_torch(origin,tensor,angle)
        return rotated

    def augment_shear(self, sign: dict, type: str, squeeze_ratio: tuple) -> dict:
        src = np.array(((0, 1), (1, 1), 
                        (0, 0), (1, 0)), dtype=np.float32)

        if type == "squeeze":
            move_left = random.uniform(*squeeze_ratio)
            move_right = random.uniform(*squeeze_ratio)

            if random.random() > 0.5:
                dest = np.array(((0 + move_left, 1), (1 - move_right, 1), 
                                 (0 + move_left, 0), (1 - move_right, 0)),
                                dtype=np.float32)
            else:
                dest = np.array(((0, 1+ move_left), (1, 1+ move_left), 
                                 (0, 0- move_right), (1, 0- move_right)),
                                dtype=np.float32)                
            mtx = cv2.getPerspectiveTransform(src, dest)

        elif type == "perspective":

            dest = np.array(((0+random.uniform(*squeeze_ratio), 1+random.uniform(*squeeze_ratio)), 
                             (1+random.uniform(*squeeze_ratio), 1+random.uniform(*squeeze_ratio)), 
                             (0+random.uniform(*squeeze_ratio), 0+random.uniform(*squeeze_ratio)),
                             (1+random.uniform(*squeeze_ratio), 0+random.uniform(*squeeze_ratio))), dtype=np.float32)

            mtx = cv2.getPerspectiveTransform(src, dest)

        else:

            logging.error("Unsupported shear type provided.")
            return {}


        augmented_landmarks = cv2.perspectiveTransform(np.array(sign.cpu(), dtype=np.float32), mtx)
        
        # reverse 0,0 in 0,0 because there are points not recognized and set 0,0
        augmented_zero_landmark = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), mtx)[0][0]
        augmented_landmarks = np.stack([np.where(sub == augmented_zero_landmark, [0, 0], sub) for sub in augmented_landmarks])

        sign = torch.from_numpy(augmented_landmarks).to(self.device)
        return sign


    def augment_arm_joint_rotate(self, sign: dict, probability: float, angle_range: tuple) -> dict:
        
        # Iterate over both directions (both hands)
        for arm_side_ids in self.ARM_IDENTIFIERS_ORDER:
            for landmark_index in range(len(arm_side_ids)-1):
                origins = sign[:,arm_side_ids[landmark_index],:]
                to_be_rotated  = [arm_side_ids[landmark_index + 1]]

                if self.__random_pass(probability):
                    angle = torch.tensor(math.radians(random.uniform(*angle_range)))
                    sign[:,to_be_rotated,:] = self.__rotate_torch(origins,sign[:,to_be_rotated,:],angle)                            
        return sign

    def get_random_transformation(self,selected_aug,depth_map_original):
        """
        depth_map = self.augment_rotate(depth_map_original, angle_range=(-23, 23))
        return depth_map
        """
        if selected_aug == 0:
            depth_map = self.augment_rotate(depth_map_original, angle_range=(-20, 20))

        if selected_aug == 1:
            depth_map = self.augment_shear(depth_map_original, "perspective", squeeze_ratio=(-0.2, 0.2))

        if selected_aug == 2:
            depth_map = self.augment_shear(depth_map_original, "squeeze", squeeze_ratio=(0.2, -0.2))

        if selected_aug == 3:
            depth_map = self.augment_arm_joint_rotate(depth_map_original, 0.5, angle_range=(-15, 15))
        return depth_map
        #"""
    
    if __name__ == "__main__":
        pass

