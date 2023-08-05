"""
Extract and save caption features locally.
"""

import os
from os import path
from os.path import join
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Parameter
# import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
import clip
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess

caption_model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True)
total_video_folders = np.sort(os.listdir('/home/da/data/Dataset/CholecT50_complete/CholecT50/videos/')) #  ['VID01', ...]
print("Num of video folders: ", len(total_video_folders))
# print("video folders: ", total_video_folders)

caption_model = caption_model.to(torch.device('cuda:0'))
text_model, _ = clip.load('ViT-L/14', device='cuda:0')
linear_model = nn.Linear(in_features = 768, out_features = 1024).to(torch.device('cuda:0'))
# vid_folders = total_video_folders[-7:]
vid_folders0 = total_video_folders[-5:]
print(vid_folders0)

def one_gpu_cap_fea(
    kf_train_folder_path: list, 
    vid_folders: list,
    cap_model, 
    tex_model, 
    li_model
):
    cunt = 0
    cap_fea_root_path = '/home/da/data/Dataset/CholecT50_complete/CholecT50/cap_fea/'
    # vid_folders = np.sort(os.listdir(kf_train_folder_path)) #  ['0000', '0001', ...]
    print(len(vid_folders))
    for t in range(len(vid_folders)):
        if not path.exists(join(cap_fea_root_path, vid_folders[t])):
            print("create new folder ...")
            os.mkdir(join(cap_fea_root_path, vid_folders[t]))
        
        os.chdir(join(cap_fea_root_path, vid_folders[t]))
        kframes = np.sort(os.listdir(join(kf_train_folder_path, vid_folders[t])))
        # cap_fea = {}
        for i in range(len(kframes)):
            ###########################
            #      Image Caption      #
            ###########################
            cap_fea = {}
            cunt += 1
            kfname = join(kf_train_folder_path, vid_folders[t], kframes[i])
            raw_image = Image.open(kfname).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(torch.device('cuda'))
            # generate caption
            caption = cap_model.generate({"image": image})
            print("frame {} caption: ".format(kframes[i]), caption)
            caption_fea = tex_model.encode_text(clip.tokenize(caption).to(torch.device('cuda'))).float()
            caption_fea = caption_fea.to(torch.device('cuda'))
            caption_fea = li_model(caption_fea)
            # caption_fea = caption_fea.to('cpu')

            # cap_fea['{}'.format(join(vid_folders[t], kframes[i]))] = caption_fea
            cap_fea['{}'.format(kframes[i])] = caption_fea

            del image
            del caption
            del caption_fea

            torch.save(cap_fea, kframes[i] + '.pt')
            del cap_fea



if __name__ == '__main__':
    # caption_fea('/home/da/data/Dataset/Vid_HOI/keyframes_train/')
    kf_path = '/home/da/data/Dataset/CholecT50_complete/CholecT50/videos/'
    one_gpu_cap_fea(kf_path, vid_folders0, caption_model, text_model, linear_model)