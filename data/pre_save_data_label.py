"""
Save data and labels on SSD, and load them directly to memory while training
"""

import numpy as np
import os
from os import listdir
from os.path import join
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image

from CholecT50_dataloader import make_video_clip

ROOT_DIR = '/home/da/Desktop/CholecT50_complete/CholecT50/'
video_folders = np.sort(listdir(join(ROOT_DIR, 'videos')))
labels = np.sort(listdir(join(ROOT_DIR, 'labels')))
# print(video_folders)  # ['VID01' 'VID02', ... ]
# print(labels)

train_split = ['01', '15', '26', '40', '52', '65', '79', '02', '18', '27', '43', '56', '66', 
                        '92', '04', '22', '31', '47', '57', '68', '96', '05', '23', '35', '48', '60', 
                        '70', '103', '13', '25', '36', '49', '62', '75', '110']
train_split1 = ['70', '110']
val_split = ['08', '12', '29', '50', '78']
test_split = ['06', '51', '10', '73', '14', '74', '32', '80', '42', '111']

transform = transforms.Compose(
            [
                transforms.Resize((256, 448)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

def pre_save(mode: str):
    if mode == 'train':
        data = train_split
    elif mode == 'train1':
        data = train_split1
    elif mode == 'val':
        data = val_split
    else:
        data = test_split
    
    real_video_folders = [f'VID{i}' for i in data]
    real_labels = [f'VID{i}.json' for i in data]
    print("mode: ", mode)
    print(real_video_folders)

    dataset = make_video_clip(ROOT_DIR, real_video_folders, real_labels)
    # print(len(dataset))  #  12585
    tool_label = np.zeros([6])
    verb_label = np.zeros([10])
    target_label = np.zeros([15])
    triplet_label = np.zeros([100])
    phase_label = np.zeros([7])

    ls = []
    for i, clip in tqdm(enumerate(dataset)):
        # clip = dataset[idx]
        frame_names = clip[0]
        # print("frame_names: ", frame_names)
        frame_labels = clip[1]
        clip_imgs, trip_labels, ins_labels, ver_labels, tar_labels, phase_labels = [], [], [], [], [], []

        for frame in frame_names:
            img_path = join(ROOT_DIR, 'videos', frame)
            tensor_img = read_image(img_path)
            tensor_img = transform(transforms.ToPILImage()(tensor_img))
            clip_imgs.append(tensor_img)

        for label in frame_labels:
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
            ins_labels.append(torch.from_numpy(tool_label))
            ver_labels.append(torch.from_numpy(verb_label))
            tar_labels.append(torch.from_numpy(target_label))
            phase_labels.append(torch.from_numpy(phase_label))
        
        sample = {
            'clip imgs': torch.stack(clip_imgs), 
            'trip labels': torch.stack(trip_labels), 
            'ins labels': torch.stack(ins_labels), 
            'ver labels': torch.stack(ver_labels), 
            'tar labels': torch.stack(tar_labels), 
            'phase labels': torch.stack(phase_labels), 
            'frame names': frame_names
        }

        # ls.append((clip_imgs, trip_labels, ins_labels, ver_labels, tar_labels, phase_labels))
        ls.append(sample)
        # print(f"clip {i} done!")

    npy_file = np.array(ls)
    np.save(join(ROOT_DIR, f'pre_saved_data_label/all_data_label_{mode}.npy'), arr = npy_file)
    print('-' * 100)


if __name__ == '__main__':
    _mode_list = ['train1']
    
    for m in _mode_list:
        pre_save(m)