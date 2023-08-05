"""
Save image features from CLIP, and load them directly to memory while training
"""

import numpy as np
import os
from os import listdir, path
from os.path import join
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn, Tensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
import clip

from CholecT50_dataloader import make_video_clip

ROOT_DIR = '/mnt/data0/Datasets/CholecT50_complete/CholecT50/'
video_folders = np.sort(listdir(join(ROOT_DIR, 'videos')))
labels = np.sort(listdir(join(ROOT_DIR, 'labels')))
# print(video_folders)  # ['VID01' 'VID02', ... ]
# print(labels)

transform = transforms.Compose(
    [
        transforms.Resize((256, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ]
)

model, preprocess = clip.load("ViT-B/32", device='cuda:0')


def pre_save():
    # dataset = make_video_clip(ROOT_DIR, video_folders, labels)
    # print(len(dataset))  #  12585

    for vid in video_folders:
        dataset = make_video_clip(ROOT_DIR, [vid])
        if not path.exists(join(ROOT_DIR, 'img_fea_CLIP_1', vid)):
            print("creating new folder ...")
            os.mkdir(join(ROOT_DIR, 'img_fea_CLIP_1', vid))

        os.chdir(join(ROOT_DIR, 'img_fea_CLIP_1', vid))
        # frames = np.sort(listdir(join(ROOT_DIR, 'videos', vid)))

        for i, clip in tqdm(enumerate(dataset)):
            frame_names = clip
            # print("frame_names: ", frame_names)
            clip_fea_dict = {}
            fea_list = []

            for frame in frame_names:
                img_path = join(ROOT_DIR, 'videos', frame)
                tensor_img = read_image(img_path)
                tensor_img = transform(transforms.ToPILImage()(tensor_img))
                image = preprocess(tensor_img).unsqueeze(0).to(device='cuda:0')
                # img_path = join(ROOT_DIR, 'videos', frame)
                # tensor_img = read_image(img_path)
                # tensor_img = transform(transforms.ToPILImage()(tensor_img))
                # clip_imgs.append(tensor_img)
            # clip_frame_fea = net(torch.stack(clip_imgs).to(device = 'cuda:0'))
                with torch.no_grad():
                    frame_fea = model.encode_image(image)
                    # print("frame_fea shape: ", frame_fea.shape)  # torch.Size([1, 512])
                    # print(clip_frame_fea.shape)  # torch.Size([8, 1024])
                    fea_list.append(frame_fea)
            clip_fea_dict[f'{frame_names[0]}'] = torch.stack(fea_list)

            torch.save(clip_fea_dict, join(
                ROOT_DIR, 'img_fea_CLIP_1', frame_names[0]) + '.pt')

        print(f"{vid} image features has been saved!")
        print('-' * 100)


if __name__ == '__main__':
    pre_save()
