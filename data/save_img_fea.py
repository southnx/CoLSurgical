"""
Save data and labels on SSD, and load them directly to memory while training
"""

import numpy as np
import os
from os import listdir, path
from os.path import join
from tqdm import tqdm
import torch
from torch import nn, Tensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights

from CholecT50_dataloader import make_video_clip

ROOT_DIR = '/home/da/Desktop/CholecT50_complete/CholecT50/'
video_folders = np.sort(listdir(join(ROOT_DIR, 'videos')))
labels = np.sort(listdir(join(ROOT_DIR, 'labels')))
# print(video_folders)  # ['VID01' 'VID02', ... ]
# print(labels)

transform = transforms.Compose(
            [
                transforms.Resize((256, 448)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

weights = ResNet18_Weights.DEFAULT
resnet1 = resnet18(weights=weights)
# resnet1.eval()
resnet1.fc = nn.Sequential(
    nn.Dropout(p=0.9),
    nn.Linear(in_features=resnet1.fc.in_features, out_features=1024))
net = resnet1.to(device = 'cuda:0')

def pre_save():
    # dataset = make_video_clip(ROOT_DIR, video_folders, labels)
    # print(len(dataset))  #  12585

    for vid in video_folders:
        dataset = make_video_clip(ROOT_DIR, [vid])
        if not path.exists(join(ROOT_DIR, 'img_fea', vid)):
            print("creating new folder ...")
            os.mkdir(join(ROOT_DIR, 'img_fea', vid))
        
        os.chdir(join(ROOT_DIR, 'img_fea', vid))
        # frames = np.sort(listdir(join(ROOT_DIR, 'videos', vid)))

        for i, clip in tqdm(enumerate(dataset)):
            # clip = dataset[idx]
            # print("clip: ", clip)
            frame_names = clip
            print("frame_names: ", frame_names)
            clip_img_fea = {}
            clip_imgs = []

            for frame in frame_names:
                img_path = join(ROOT_DIR, 'videos', frame)
                tensor_img = read_image(img_path)
                tensor_img = transform(transforms.ToPILImage()(tensor_img))
                clip_imgs.append(tensor_img)
            clip_frame_fea = net(torch.stack(clip_imgs).to(device = 'cuda:0'))
            # print(clip_frame_fea.shape)  # torch.Size([8, 1024])
            clip_img_fea[f'{frame_names[0]}'] = clip_frame_fea

            torch.save(clip_img_fea, join(ROOT_DIR, 'img_fea', frame_names[0]) + '.pt')

        print(f"{vid} image features has been saved!")
        print('-' * 100)


if __name__ == '__main__':
    pre_save()