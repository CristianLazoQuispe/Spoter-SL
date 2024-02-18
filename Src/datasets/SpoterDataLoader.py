from torch.utils.data import DataLoader
import math 
import random
import torch
import copy

class SpoterDataLoader(DataLoader):
    def __init__(self, dataset, shuffle=True, **kwargs):
        super(SpoterDataLoader, self).__init__(dataset, shuffle=shuffle, **kwargs)

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
        indices = []
        try:
            for _ in range(self.batch_size):
                indices.append(next(self.index_sampler))
        except StopIteration:
            if len(indices)==0:
                return None,None,None
                
        depth_maps = [None for i in range(len(indices))]
        labels = [None for i in range(len(indices))]
        video_names = [None for i in range(len(indices))]
        for i,idx in enumerate(indices):
            if self.dataset.has_augmentation:
                idx = self.dataset.map_ids_augmentation[idx]
                
            depth_map = torch.from_numpy(self.dataset.data[idx]).to('cuda')

    
            video_name = self.dataset.video_name[idx].decode('utf-8')
            label = torch.Tensor([self.dataset.labels[idx]])
    
            if self.dataset.transform:
                # Apply potential augmentations
                if random.random() > self.dataset.augmentations_prob:        
                    n_aug = random.randrange(4)+1 #[1,2,3,4]
                    for j in range(n_aug):
                        selected_aug = random.randrange(4)
                        depth_map = self.dataset.augmentation.get_random_transformation(selected_aug,depth_map)

                depth_map = self.dataset.transform(depth_map)

            depth_map = depth_map - 0.5
                
            depth_maps[i] = depth_map.to('cuda')
            labels[i] = label.to('cuda', dtype=torch.long)
            video_names[i] = video_name

        return depth_maps, labels, video_names


############## SIN AUGMENTATION ####################
#Train Epoch 21:   100%|████████████████████| 12/12 [00:08<00:00,  1.42it/s, id_aug=40, m_acc=0.0591, m_loss=4.25]
#Val   Epoch 21:   100%|████████████████████| 3/3 [00:00<00:00,  3.51it/s, id_aug=59, m_acc=0.118, m_loss=4.16] 
############## CON AUGMENTATION factor 2 ####################
#Train Epoch 7:    100%|████████████████████| 44/44 [00:28<00:00,  1.56it/s, id_aug=21, m_acc=0.0573, m_loss=3.9] 
#Val   Epoch 7:    100%|████████████████████| 3/3 [00:00<00:00,  3.97it/s, id_aug=59, m_acc=0.0642, m_loss=4.18]