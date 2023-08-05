"""
ACoLP model
"""

from os import path
from os.path import join
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import Parameter
# import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights
import clip
import numpy as np
from PIL import Image
from lavis.models import load_model_and_preprocess
from timm.models.layers import trunc_normal_

from .dynamic_gnn import DynamicGNN
from .HOIPrompting import HOIPrompt 
from .mlp import MLP
from .Action_Prompt import ActionPrompt


def remove(string):
    return string.replace(" ", "")

def remove2(string):
    return string.replace("\n", "")

text_model, text_preprocess = clip.load('ViT-L/14', device='cpu')

class ACoLP(nn.Module):
    def __init__(
        self, 
        num_proposals: int = 15, 
        roi_map_size: int = 32, 
        node_dim: int = 1024, 
        is_prompt: bool = True, 
        output_layer = 'layer4', 
        triplet_file: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/HOI.txt', 
        hoi_prompt: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/hoi_prompt.pt', 
        act_prompt: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/act_prompt.pt', 
        caption_feas: str = '/home/da/data/Dataset/CholecT50_complete/CholecT50/cap_fea/',
        ROOT_TRAIN: str = '/home/da/data/Dataset/TripletAction_ACoLP/small_keyframes_train/', 
        ROOT_VAL: str = '/home/da/data/Dataset/TripletAction_ACoLP/small_keyframes_val/'
    ):
        super().__init__()
        self.num_proposals = num_proposals
        self.roi_map_size = roi_map_size
        self.node_dim = node_dim
        self.is_prompt = is_prompt
        self.triplet_file = triplet_file
        self.dim_visual_fea = 256
        self.dim_prompt_fea = 512
        self.hoi_prompt = hoi_prompt
        self.act_prompt = act_prompt
        self.caption_feas = caption_feas
        self.ROOT_TRAIN = ROOT_TRAIN
        self.linear = nn.Linear(in_features = 768, out_features = self.node_dim) 
        self.caphoi_prompt = HOIPrompt(embed_dim = self.node_dim) 
        self.vishoi_prompt = HOIPrompt(embed_dim = self.node_dim) 
        self.mlp_pred = MLP(2 * self.node_dim, self.node_dim // 4, 1, 3) 
        self.act_prompt_net = ActionPrompt()

        # ResNet model for feature extraction
        # resnet1 = models.resnet18(pretrained=True)
        weights = ResNet18_Weights.DEFAULT
        resnet1 = resnet18(weights=weights)
        # resnet1.eval()
        resnet1.fc = nn.Sequential(
            nn.Dropout(p=0.9),
            nn.Linear(in_features=resnet1.fc.in_features, out_features=1024))
        self.net = resnet1

        # generate HOI prompts
        self.HOIs = []
        with open(self.triplet_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                l = remove2(lines[i])
                # l = remove2(l)
                self.HOIs.append(l)
        self.num_hois = len(self.HOIs)  # 100
        print("num hois: ", self.num_hois)
        self.all_text_fea = torch.zeros(1, self.node_dim) 
        # self.all_text_fea = {}
        if not path.exists(self.hoi_prompt):
            for i in range(self.num_hois):
                ins, verb, tar = self.HOIs[i].split(",")[0], self.HOIs[i].split(",")[1], self.HOIs[i].split(",")[2]
                text_fea = text_model.encode_text(clip.tokenize(f'A physician is using {ins} to\
                     {verb} the {tar}in an endoscopic surgery.')).float()
                text_fea = self.linear(text_fea)
                # print("text_fea shape: ", text_fea.shape)  # torch.Size([1, 1024])
                self.all_text_fea = torch.cat((self.all_text_fea, text_fea), dim = 0)
            self.all_text_fea = self.all_text_fea[1:]
            torch.save(self.all_text_fea, self.hoi_prompt)
            # self.all_text_fea = torch.load(self.hoi_prompt)
            self.all_text_fea = nn.Parameter(torch.load(self.hoi_prompt), requires_grad=False)
        else:
            self.all_text_fea = nn.Parameter(torch.load(self.hoi_prompt), requires_grad=False)
        # print("self.all_text_feature: ", self.all_text_fea)  # torch.Size([100, 1024])
        # print("HOI feature shape: ", self.all_text_fea.shape) # torch.Size([557, 1024])
        # print("HOI feature: ", self.all_text_fea) 
        # {'person,grab,cat': tensor([[-0.2907, -0.2603,  0.0458,  ..., -0.1754, -0.0007,  0.3003]], ...}
        # print("Number of HOIs: ", len(all_text_fea)) # 557

        # generate action features
        self.actions = ['grasping', 'retracting', 'dissecting', 'coagulating', 'clipping', 'cutting', 
                   'aspirating', 'irrigating', 'packing', 'null_verb']
        self._actions = ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 
                   'aspirate', 'irrigate', 'pack', 'null_verb']
        print("action num: ", len(self.actions))
        self.all_action_fea = torch.zeros(1, self.node_dim) 
        if not path.exists(self.act_prompt):
            num_act = len(self.actions)  # 50
            for i in range(num_act):
                act_prompt = text_model.encode_text(clip.tokenize(f'A physician is {self.actions[i]} ...')).float()
                act_prompt = self.linear(act_prompt)
                self.all_action_fea = torch.cat((self.all_action_fea, act_prompt), dim = 0)
            self.all_action_fea = self.all_action_fea[1:]
            torch.save(self.all_action_fea, self.act_prompt)
            # self.all_action_fea = torch.load(self.act_prompt)
            self.all_action_fea = nn.Parameter(torch.load(self.act_prompt), requires_grad=False)
        else:
            self.all_action_fea = nn.Parameter(torch.load(self.act_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([10, 1024])

        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, video_seg: list, frame_names: list):
        r"""
        video_seg: video frames in a batch of video clips    torch.Size([40, 8, 3, 256, 448])
        frame_names: frame names in a batch of video clips   torch.Size([40, 8])
        """
        video_fea = {}
        # num_frames = len(video_seg)
        # print("video_seg shape: ", video_seg.shape)  # torch.Size([8, 3, 256, 448])
        num_frames = video_seg.shape[0]
        # print("num frames: ", num_frames)  # 8
        # print(len(frame_names))  # 8
        # print("Video seg shape: ", video_seg[1].shape)  # torch.Size([10, 3, 256, 448])
        # assert len(video_seg) == len(frame_names)

        # single_frmae_name = frame_names[i]
        # print("frame_names: ", frame_names)  # ['VID23/001056.png', ... ]
        clip_vis_prompt_fea_list = []
        clip_learned_action_prompt_list = []  # the learned action prompts in a video clip
        for i in range(num_frames):
            folder_name = frame_names[i].split('/')[0]  # 'VID08'
            frame_name = frame_names[i].split('/')[1]   # '000800.png"
            caption_fea = nn.Parameter(torch.load(join(
                self.caption_feas, folder_name, frame_name + '.pt'))[f'{frame_name}'], requires_grad=False)
            # print("caption feature: ", caption_fea)  # torch.Size([1, 1024])

            # ------ CaptionHOI Prompting ------ #
            # Each frame has its unique caption prompt feature
            cap_prompt_fea = self.caphoi_prompt(self.all_text_fea.unsqueeze(0), caption_fea.unsqueeze(0))
            # print("cap_prompt_fea shape: ", cap_prompt_fea.shape)  # torch.Size([1, 100, 1024])
            del caption_fea

            ###########################
            # Extract visual features #
            ###########################
            # frame_fea = resnet_model(video_seg[i].unsqueeze(0))   # on cpu !
            frame_fea = self.net(video_seg[i].unsqueeze(0))   # on cpu !
            # print("frame feature: ", frame_fea)
            # print("frame feature shape: ", frame_fea.shape)  # torch.Size([1, 1024])
            # print("frame feature: ", frame_fea) 

            # ------ VisualHOI Prompting ------ #
            vis_prompt_fea = self.vishoi_prompt(cap_prompt_fea, frame_fea.unsqueeze(0))
            # print("vis_prompt_fea shape: ", vis_prompt_fea.shape) # torch.Size([1, 100, 1024])
            # print("vis_prompt_fea: ", vis_prompt_fea)
            clip_vis_prompt_fea_list.append(vis_prompt_fea) # HOI prompt features
            # del clip_edge_fea_list
            # print("clip_vis_prompt_fea_list: ", clip_vis_prompt_fea_list)

            # Action Promprt Network
            learned_action_prompt = self.act_prompt_net(
                frame_fea.unsqueeze(0), self.all_action_fea.unsqueeze(0))
            # print("learned_action_prompt: ", learned_action_prompt.shape) # torch.Size([10, 1024])
            clip_learned_action_prompt_list.append(learned_action_prompt)
        # print("clip_vis_prompt_fea_list shape: ", clip_vis_prompt_fea_list[0].shape) # torch.Size([1, 100, 1024])
        # print("clip_vis_prompt_fea_list shape: ", len(clip_vis_prompt_fea_list)) # 8

        ###########################
        #       Dynamic GNN       #
        ########################### 
        # update action prompts
        num_actions = len(self.actions)
        adj_matrix = torch.ones((num_actions, num_actions))
        dynamic_gnn_model = DynamicGNN(dim_in = self.node_dim, dim_out = self.node_dim, adj = adj_matrix)
        # print("clip_learned_action_prompt_list shape: ", len(clip_learned_action_prompt_list)) 
        # print("num frames: ", num_frames) # 8
        N = len(clip_learned_action_prompt_list)
        assert N == num_frames
        if N >= 2:
            for i in range(1, N):
                update_learned_act_prompt = dynamic_gnn_model(
                    clip_learned_action_prompt_list[i], clip_learned_action_prompt_list[i-1]
                )
                clip_learned_action_prompt_list[i] = update_learned_act_prompt
                del update_learned_act_prompt
        # print("clip_learned_action_prompt_list shape: ", clip_learned_action_prompt_list[0].shape) # torch.Size([10, 1024])
        # print("clip_learned_action_prompt_list: ", clip_learned_action_prompt_list) # 8
        # print("clip_learned_action_prompt_list: ", len(clip_learned_action_prompt_list)) # 8

        ##################################
        #       Triplet prediction       #
        ##################################
        clip_trip_pred_list = []
        for i in range(num_frames):
            triplet_pred = torch.zeros(self.num_hois)
            for j in range(self.num_hois):
                action_name = self.HOIs[j].split(",")[1]
                act_p = self.all_action_fea[self._actions.index(action_name)]
                hoi_p = self.all_text_fea[j]
                triplet_pred[j] = torch.sigmoid(self.mlp_pred(torch.cat((act_p, hoi_p), dim=0)))
            clip_trip_pred_list.append(triplet_pred)
            # print("triplet_pred shape: ", triplet_pred.shape)  # torch.Size([100])
            # print("triplet_pred: ", triplet_pred)
        # print("clip_trip_pred_list: ", len(clip_trip_pred_list))  # 8

        return clip_trip_pred_list