from torch.utils.data import DataLoader
import math 
import random
import torch
import copy

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
            for i in range(int(self.dataset.factors[label.item()]+0.5)):
                # Apply potential augmentations
                depth_map = copy.deepcopy(depth_map_original)
                if random.random() < self.dataset.augmentations_prob:
                    n_aug = random.randrange(4)+1 #[1,2,3,4]
                    #print("n_aug:",n_aug)
                    for j in range(n_aug):
                        selected_aug = random.randrange(4)
                        depth_map = self.dataset.augmentation.get_random_transformation(selected_aug,depth_map)

                depth_map = depth_map - 0.5
                if self.dataset.transform:
                    # Gaussian Noise
                    depth_map = self.dataset.transform(depth_map)

                sample = (depth_map, label, video_name)
                batch.append(sample)

        depth_maps, labels, video_names = zip(*batch)
        return torch.stack(depth_maps), torch.stack(labels), video_names
