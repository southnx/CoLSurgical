"""
Load CholecT50 dataset in the form of video clips (segments), not single frames
"""

import os
from os.path import join
from os import listdir
import json
import random
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.utils.data.dataloader import default_collate


class CholecT50(Dataset):
    def __init__(self, root_dir, mode: str):
        super(CholecT50, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        if self.mode == 'train':
            self.npy = join(self.root_dir, 'pre_saved_data_label/all_data_label_train.npy')
            self.npy = np.load(self.npy, allow_pickle=True)
        elif self.mode == 'train1':
            self.npy = join(self.root_dir, 'pre_saved_data_label/all_data_label_train1.npy')
            self.npy = np.load(self.npy, allow_pickle=True)
        elif self.mode == 'val':
            self.npy = join(self.root_dir, 'pre_saved_data_label/all_data_label_val.npy')
            self.npy = np.load(self.npy, allow_pickle=True)
        else:
            self.npy = join(self.root_dir, 'pre_saved_data_label/all_data_label_test.npy')
            self.npy = np.load(self.npy, allow_pickle=True)

    def __len__(self):
        return len(self.npy) # num of video clips in train / val / test

    def __getitem__(self, idx):
        """
        idx: the index of video clips in the whole dataset
        """
        sample = self.npy[idx]

        return sample


def collate_fn_vid(batch):
    # print("batch: ", batch)  # batch_size
    size = len(batch)
    # print("batch size: ", size)
    batch_imgs, batch_trip_labels, batch_frame_names = [], [], []
    for i in range(size):
        batch_imgs.append(batch[i]['clip imgs'])
        batch_trip_labels.append(batch[i]['trip labels'])
        batch_frame_names.append(batch[i]['frame names'])
    
    return torch.stack(batch_imgs), torch.stack(batch_trip_labels), batch_frame_names
    

if __name__ == "__main__":
    ChoT50 = CholecT50(
        root_dir = '/home/da/data/Dataset/CholecT50_complete/CholecT50/', 
        mode = 'train', 
        train_split = ['01', '15', '26', '40', '52', '65', '79', '02', '18', '27', '43', '56', '66', 
                        '92', '04', '22', '31', '47', '57', '68', '96', '05', '23', '35', '48', '60', 
                        '70', '103', '13', '25', '36', '49', '62', '75', '110'], 
        val_split = ['08', '12', '29', '50', '78'],
        test_split = ['06', '51', '10', '73', '14', '74', '32', '80', '42', '111']
    )
    print(ChoT50)