from torch.utils.data import DataLoader
import math 
import random
import torch

class AugmentedDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=True, **kwargs):
        super(AugmentedDataLoader, self).__init__(dataset, shuffle=shuffle, **kwargs)

    def __iter__(self):
        return AugmentedDataLoaderIterator(self)

class AugmentedDataLoaderIterator:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = loader.dataset
        self.batch_size = loader.batch_size
        self.index_sampler = iter(loader.sampler)

    def __iter__(self):
        return self

    def __next__(self):
        batch = []
        try:
            indices = [next(self.index_sampler) for _ in range(self.batch_size)]
        except StopIteration:
            return None,None,None

        for idx in indices:
            sample = self.dataset[idx]
            depth_map_original, label, video_name = sample

            #depth_map_original = depth_map.to("cpu")
            for cnt in range(int(self.dataset.factors[label.item()]+0.5)):
                # Apply potential augmentations
                depth_map = depth_map_original
                if random.random() < self.dataset.augmentations_prob:
                    selected_aug = random.randrange(4)
                    if selected_aug == 0:
                        depth_map = self.dataset.augmentation.augment_rotate(depth_map_original, angle_range=(-13, 13))

                    if selected_aug == 1:
                        depth_map = self.dataset.augmentation.augment_shear(depth_map_original, "perspective", squeeze_ratio=(0, 0.1))

                    if selected_aug == 2:
                        depth_map = self.dataset.augmentation.augment_shear(depth_map_original, "squeeze", squeeze_ratio=(0, 0.15))

                    if selected_aug == 3:
                        depth_map = self.dataset.augmentation.augment_arm_joint_rotate(depth_map_original, 0.3, angle_range=(-4, 4))

                depth_map = depth_map - 0.5
                #depth_map = depth_map.to('cuda')
                if self.dataset.transform:
                    depth_map = self.dataset.transform(depth_map)
                    
                #depth_map = depth_map.to('cuda')
                #label = label.to('cuda', dtype=torch.long)

                sample = (depth_map, label, video_name)
                batch.append(sample)

        depth_maps, labels, video_names = zip(*batch)
        return torch.stack(depth_maps), torch.stack(labels), video_names
