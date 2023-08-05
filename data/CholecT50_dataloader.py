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


def make_video_clip(root_dir, video_folders: list, label_files: list = None, num_frames = 8):
    """
    generate video clips for all the videos of train / val / test
    """
    dataset = []
    for i in range(len(video_folders)):
        frames = np.sort(listdir(join(root_dir, 'videos', video_folders[i])))
        frame_names = [join(video_folders[i], frames[j]) for j in range(len(frames))]
        num_clips = np.ceil(len(frame_names) / num_frames)
        if label_files:
            labels = json.load(open(join(root_dir, 'labels', label_files[i]), 'rb'))["annotations"]
            # labels = torch.from_numpy(labels)
        for t in range(int(num_clips)):
            vid_label = []
            if (t+1) * num_frames <= len(frame_names):
                vid_clip_name = frame_names[t * num_frames: (t+1) * num_frames]
                if label_files:
                    for k in range(num_frames):
                        vid_label.append(labels[f'{t * num_frames + k}'])
                    dataset.append((vid_clip_name, vid_label))
                else:
                    dataset.append((vid_clip_name))
            # else:
            #     vid_clip_name = frame_names[t * num_frames : ]
            #     for k in range(len(frame_names) - t * num_frames):
            #         vid_label.append(labels[f'{t * num_frames + k}'])
            #     dataset.append((vid_clip_name, vid_label))

    return dataset


class CholecT50(Dataset):
    def __init__(self, root_dir, mode: str, train_split: list, val_split: list, test_split: list, 
                transform = None):
        super(CholecT50, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.data = train_split
        elif self.mode == 'val':
            self.data = val_split
        else:
            self.data = test_split
        # print("self.data: ", self.data)  # ['01', '15', ... ]

        self.video_folders = np.sort(listdir(join(self.root_dir, 'videos')))
        self.label_files = np.sort(listdir(join(self.root_dir, 'labels')))
        assert len(self.video_folders) == len(self.label_files)
        print(self.video_folders)

        # self.real_video_folders = [self.video_folders[i] for i in self.data]
        self.real_video_folders = [f'VID{i}' for i in self.data]
        # self.real_label_files = [self.label_files[i] for i in self.data]
        self.real_label_files = [f'VID{i}.json' for i in self.data]
        # print("real video folders: ", self.real_video_folders)
        # print("real label files: ", self.real_label_files)
        self.real_data = make_video_clip(self.root_dir, self.real_video_folders, self.real_label_files)
        print("Num of video clips: ", len(self.real_data))
        print('-' * 50)
        # print("real data example: ", self.real_data[100]) # 2  (['VID08/000800.png', ...], [])
        # print(len(self.real_data[100][1])) # 8

    def __len__(self):
        return len(self.real_data) # num of video clips in train / val / test

    def __getitem__(self, idx):
        """
        idx: the index of video clips in the whole dataset
        """
        tool_label = np.zeros([6])
        verb_label = np.zeros([10])
        target_label = np.zeros([15])
        triplet_label = np.zeros([100])
        phase_label = np.zeros([7])

        clip_data = self.real_data[idx]
        frame_names, labels = clip_data[0], clip_data[1]
        # print("frames: ", frame_names)  # 8
        # print("labels: ", labels)
        clip_imgs, trip_labels = [], []
        # for i in range(len(frames)):
        for frame in frame_names:
            img_path = join(self.root_dir, 'videos', frame)
            # print("img_path: ", img_path)
            # img = Image.open(img_path)
            # t_img = torch.Tensor(np.array(img))
            # t_img = torch.permute(t_img, (2, 0, 1))
            t_img = read_image(img_path)
            # print("img shape: ", t_img.shape)  # torch.Size([3, 480, 854])
            if self.transform:
                t_img = self.transform(transforms.ToPILImage()(t_img))
            # print("img shape: ", t_img.shape)  # torch.Size([3, 256, 448])
            clip_imgs.append(t_img)
        for label in labels:
            for j in range(len(label)):
                triplet = label[j][0:1]
                if triplet[0] != -1.0:
                    # triplet_label[triplet[0]] += 1
                    triplet_label[triplet[0]] = 1
                tool = label[j][1:7]
                if tool[0] != -1.0:
                    # tool_label[tool[0]] += 1
                    tool_label[tool[0]] = 1
                verb = label[j][7:8]
                if verb[0] != -1.0:
                    # verb_label[verb[0]] += 1
                    verb_label[verb[0]] = 1
                target = label[j][8:14]  
                if target[0] != -1.0:   
                    # target_label[target[0]] += 1   
                    target_label[target[0]] = 1       
                phase = label[j][14:15]
                if phase[0] != -1.0:
                    # phase_label[phase[0]] += 1
                    phase_label[phase[0]] = 1
            # clip_labels.append((triplet_label, tool_label, verb_label, target_label, phase_label))
            trip_labels.append(torch.from_numpy(triplet_label))
        # print("clip_imgs: ", len(clip_imgs))  # 8
        # print("clip_labels: ", len(clip_labels))
        sample = {'clip imgs': torch.stack(clip_imgs), 
                  'trip labels': torch.stack(trip_labels), 
                  'frame names': frame_names}
        # print("sample: ", sample['frame names'])
        # print("frame names: ", frame_names)  # ['VID75/000616.png', ... ]
        
        # return clip_imgs, clip_labels, frame_names
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