"""
ACoLP model new
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
import time

from .dynamic_gnn import DynamicGNN
from .HOIPrompting import HOIPrompt
from .mlp import MLP
from .Action_Prompt import ActionPrompt
from models.args import get_args
from utils.fea_visualization import draw_fea

args = get_args()
device = torch.device('cuda', args.local_rank)


def remove(string):
    return string.replace(" ", "")


def remove2(string):
    return string.replace("\n", "")


text_model, text_preprocess = clip.load('ViT-L/14', device='cpu')


class ACoLP(nn.Module):
    def __init__(
        self,
        node_dim: int = 1024,
        triplet_file: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/HOI.txt', 
        hoi_prompt: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/hoi_prompt.pt', 
        act_prompt: str = '/home/da/Desktop/Code/TripletAction_ACoLP/utils/act_prompt.pt', 
        caption_feas: str = '/home/da/Desktop/CholecT50_complete/CholecT50/cap_fea/',
        img_feas: str = '/home/da/Desktop/CholecT50_complete/CholecT50/img_fea/'
    ):
        super().__init__()
        self.node_dim = node_dim
        self.triplet_file = triplet_file
        self.dim_visual_fea = 256
        self.dim_prompt_fea = 512
        self.hoi_prompt = hoi_prompt
        self.act_prompt = act_prompt
        self.caption_feas = caption_feas
        self.img_feas = img_feas
        self.linear = nn.Linear(in_features=768, out_features=self.node_dim)
        self.caphoi_prompt = HOIPrompt(embed_dim=self.node_dim)
        self.vishoi_prompt = HOIPrompt(embed_dim=self.node_dim)
        self.mlp_pred = MLP(2 * self.node_dim, self.node_dim//4, 1, 3)  #
        self.mlp_pred_act = MLP(self.node_dim, self.node_dim//4, 1, 3)  #
        # self.mlp_img_fea = nn.Linear(self.node_dim, self.node_dim)
        self.act_prompt_net = ActionPrompt()
        self.adj_matrix = torch.ones((10, 10))
        self.dynamic_gnn_model = DynamicGNN(dim_in=self.node_dim,
                                            dim_out=self.node_dim, adj=self.adj_matrix)

        # generate HOI prompts
        self.HOIs = []
        with open(self.triplet_file) as f:
            lines = f.readlines()
            for i in range(len(lines)):
                l = remove2(lines[i])
                # l = remove2(l)
                self.HOIs.append(l)
        self.num_hois = len(self.HOIs)  # 100
        # print("num hois: ", self.num_hois)
        self.all_text_fea = torch.zeros(1, self.node_dim)
        # self.all_text_fea = {}
        if not path.exists(self.hoi_prompt):
            for i in range(self.num_hois):
                ins, verb, tar = self.HOIs[i].split(",")[0], self.HOIs[i].split(",")[
                    1], self.HOIs[i].split(",")[2]
                text_fea = text_model.encode_text(clip.tokenize(f'A physician is using {ins} to\
                     {verb} the {tar}in an endoscopic surgery.')).float()
                text_fea = self.linear(text_fea)
                # print("text_fea shape: ", text_fea.shape)  # torch.Size([1, 1024])
                self.all_text_fea = torch.cat(
                    (self.all_text_fea, text_fea), dim=0)
            self.all_text_fea = self.all_text_fea[1:]
            torch.save(self.all_text_fea, self.hoi_prompt)
            # self.all_text_fea = torch.load(self.hoi_prompt)
            self.all_text_fea = nn.Parameter(
                torch.load(self.hoi_prompt), requires_grad=False)
        else:
            self.all_text_fea = nn.Parameter(
                torch.load(self.hoi_prompt), requires_grad=False)
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
        # print("action num: ", len(self.actions))
        self.all_action_fea = torch.zeros(1, self.node_dim)
        if not path.exists(self.act_prompt):
            num_act = len(self.actions)  # 50
            for i in range(num_act):
                act_prompt = text_model.encode_text(clip.tokenize(
                    f'A physician is {self.actions[i]} ...')).float()
                act_prompt = self.linear(act_prompt)
                self.all_action_fea = torch.cat(
                    (self.all_action_fea, act_prompt), dim=0)
            self.all_action_fea = self.all_action_fea[1:]
            torch.save(self.all_action_fea, self.act_prompt)
            # self.all_action_fea = torch.load(self.act_prompt)
            self.all_action_fea = nn.Parameter(
                torch.load(self.act_prompt), requires_grad=False)
        else:
            self.all_action_fea = nn.Parameter(
                torch.load(self.act_prompt), requires_grad=False)
        # print("all_action_fea: ", self.all_action_fea.shape)  # torch.Size([10, 1024])

        # get the indexes of each action in HOIs
        # {'grasp': [], 'retract': [], ... }
        # self.act_idx = {}
        # for i in range(len(self._actions)):
        #     idx = []
        #     act_name = self._actions[i]
        #     for j in range(len(self.HOIs)):
        #         verb = self.HOIs[j].split(",")[1]
        #         if verb == act_name:
        #             idx.append(j)
        #     self.act_idx[f'{act_name}'] = idx
        # print("action indexes: ", self.act_idx)
        self.act_idx = {'grasp': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 38, 39, 40],
                        'retract': [14, 15, 16, 17, 18, 19, 20, 21, 41, 42, 43, 44, 45, 63, 64, 91, 92, 93],
                        'dissect': [0, 1, 2, 32, 33, 34, 35, 36, 37, 56, 57, 58, 59, 60, 61, 62, 74, 75,
                                    76, 83, 84, 85, 86, 87],
                        'coagulate': [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 46, 47, 48, 49, 50, 51, 52, 53, 65],
                        'clip': [77, 78, 79, 80, 81], 'cut': [54, 55, 66, 67, 68, 69, 70, 71, 72, 73],
                        'aspirate': [82], 'irrigate': [88, 89, 90], 'pack': [13],
                        'null_verb': [94, 95, 96, 97, 98, 99]}

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, frame_names: list, video_seg: list = None):
        r"""
        video_seg: video frames in a batch of video clips    torch.Size([40, 8, 3, 256, 448])
        frame_names: frame names in a batch of video clips   torch.Size([40, 8])
        """
        # all_text_fea = nn.Parameter(torch.load(self.hoi_prompt), requires_grad=False)
        # all_action_fea = nn.Parameter(torch.load(self.act_prompt), requires_grad=False)
        # print("all_text_fea: ", all_text_fea)
        # num_frames = len(video_seg)
        # print("video_seg shape: ", video_seg.shape)  # torch.Size([40, 8, 3, 256, 448])

        # num_clips, num_frames = video_seg.shape[0], video_seg.shape[1]  # 40, 8
        num_clips = len(frame_names)
        num_frames = len(frame_names[0])
        # print("num frames: ", num_frames)  # 8
        # print(len(frame_names))  # 8
        # print("Video seg shape: ", video_seg[1].shape)  # torch.Size([10, 3, 256, 448])
        # assert len(video_seg) == len(frame_names)

        # single_frmae_name = frame_names[i]
        # print("frame_names: ", frame_names)  # ['VID23/001056.png', ... ]
        # clip_vis_prompt_fea_list = []
        batch_trip_pred, batch_act_pred = [], []
        for k in range(num_clips):
            # the learned action prompts in a video clip
            clip_learned_action_prompt_list = []
            # clip_vis_prompt_fea_list = []
            cap_fea_list = []
            hoi_text_fea_list = []
            time1 = time.time()
            draw_fea_list = []
            for i in range(num_frames):
                folder_name = frame_names[k][i].split('/')[0]  # 'VID08'
                frame_name = frame_names[k][i].split('/')[1]   # '000800.png"
                # caption_fea = nn.Parameter(torch.load(join(
                #     self.caption_feas, folder_name, frame_name + '.pt'))[f'{frame_name}'], requires_grad=False)
                caption_fea = torch.load(join(
                    self.caption_feas, folder_name, frame_name + '.pt'))[f'{frame_name}']
                # print("caption åfeature: ", caption_fea)  # torch.Size([1, 1024])   cuda:0
                cap_fea_list.append(caption_fea)
                # hoi_text_fea_list.append(self.all_text_fea)
                hoi_text_fea_list.append(self.all_text_fea)
                del caption_fea
            clip_cap_fea = torch.stack(cap_fea_list).to(
                device)  # torch.Size([8, 1, 1024])
            # print("clip_cap_fea: ", clip_cap_fea)
            hoi_text_fea = torch.stack(hoi_text_fea_list).to(
                device)  # torch.Size([8, 100, 1024])
            # print("hoi_text_fea: ", hoi_text_fea[0])

            # ------ CaptionHOI Prompting ------ #
            # Each frame has its unique caption prompt feature
            cap_prompt_fea = self.caphoi_prompt(hoi_text_fea, clip_cap_fea)
            # print("cap_prompt_fea shape: ", cap_prompt_fea.shape)  # torch.Size([8, 100, 1024])
            # fea_st1 = cap_prompt_fea
            # draw_fea_list.append(fea_st1)

            # time2 = time.time()
            # if args.local_rank == 0:
            #     print("section 1 takes: ", time2 - time1)

            ###########################
            # Extract visual features #
            ###########################
            # frame_fea = self.net(video_seg[k])   # on cpu !
            folder_name = frame_names[k][0].split('/')[0]  # 'VID08'
            frame_name = frame_names[k][0].split('/')[1]   # '000800.png"
            frame_fea = torch.load(join(self.img_feas, folder_name, frame_name + '.pt')
                                   )[f'{join(folder_name, frame_name)}'].type('torch.FloatTensor').to(device)
            # frame_fea = self.mlp_img_fea(frame_fea).squeeze(1).to(device)
            # print("frame feature shape: ", frame_fea.shape)  # torch.Size([8, 1024])
            # print("frame feature: ", frame_fea)

            # ------ VisualHOI Prompting ------ #
            vis_prompt_fea = self.vishoi_prompt(
                cap_prompt_fea, frame_fea.unsqueeze(1))
            # print("vis_prompt_fea shape: ", vis_prompt_fea[0].shape) # torch.Size([8, 100, 1024])
            # print("vis_prompt_fea: ", vis_prompt_fea)
            # clip_vis_prompt_fea_list.append(vis_prompt_fea) # HOI prompt features
            # del clip_edge_fea_list
            # print("clip_vis_prompt_fea_list: ", clip_vis_prompt_fea_list)
            # fea_st2 = vis_prompt_fea
            # draw_fea_list.append(fea_st2)

            # time3 = time.time()
            # if args.local_rank == 0:
            #     print("section 2 takes: ", time3 - time2)

            # ------ Action Promprt Network ------ #
            for i in range(num_frames):
                learned_action_prompt = self.act_prompt_net(
                    frame_fea[i].unsqueeze(0).unsqueeze(1), self.all_action_fea.unsqueeze(0))
                # print("learned_action_prompt: ", learned_action_prompt.shape) # torch.Size([10, 1024])
                clip_learned_action_prompt_list.append(learned_action_prompt)
            # clip_learned_action_prompt_ten = torch.stack(clip_learned_action_prompt_list)
            # print("clip_learned_action_prompt: ", clip_learned_action_prompt_ten.shape) # torch.Size([8, 10, 1024])
            # print("clip_vis_prompt_fea_list shape: ", clip_vis_prompt_fea_list[0].shape) # torch.Size([8, 100, 1024])
            # print("clip_vis_prompt_fea_list shape: ", len(clip_vis_prompt_fea_list)) # 8
            # fea_vs1 = clip_learned_action_prompt_list
            # draw_fea_list.append(fea_vs1)
            act_fea = torch.stack(clip_learned_action_prompt_list)

            # time4 = time.time()
            # if args.local_rank == 0:
            #     print("section 3 takes: ", time4 - time3)

            ###########################
            #       Dynamic GNN       #
            ###########################
            # update action prompts
            N = vis_prompt_fea.shape[0]
            # print("N: ", N)
            # assert N == num_frames
            if args.use_dgnn:
                if N >= 2:
                    for i in range(1, N):
                        update_learned_act_prompt = self.dynamic_gnn_model(
                            clip_learned_action_prompt_list[i], clip_learned_action_prompt_list[i-1]
                        )
                        clip_learned_action_prompt_list[i] = update_learned_act_prompt
                        del update_learned_act_prompt
            # print("clip_learned_action_prompt_list shape: ", clip_learned_action_prompt_list[0].shape) # torch.Size([10, 1024])
            # print("clip_learned_action_prompt_list: ", clip_learned_action_prompt_list) # 8
            # print("clip_learned_action_prompt_list: ", len(clip_learned_action_prompt_list)) # 8
            # fea_vs2 = clip_learned_action_prompt_list
            # draw_fea_list.append(fea_vs2)
            # draw_fea(draw_fea_list, frame_names[k])
            dy_act_fea = torch.stack(clip_learned_action_prompt_list)

            # time5 = time.time()
            # if args.local_rank == 0:
            #     print("section 4 takes: ", time5 - time4)

            ##################################
            #       Triplet prediction       #
            ##################################
            clip_trip_pred_list = []
            for i in range(num_frames):
                frame_act_p = clip_learned_action_prompt_list[i]
                _frame_act_p = torch.zeros_like(vis_prompt_fea[0])
                for act in self._actions:
                    idx = self.act_idx[act]
                    for j in idx:
                        _frame_act_p[j] = frame_act_p[self._actions.index(act)]
                # print("_frame_act_p: ", _frame_act_p.shape)  # torch.Size([100, 1024])
                combined_fea = torch.cat(
                    (_frame_act_p, vis_prompt_fea[i]), dim=1)
                # print("combined_fea shape; ", combined_fea.shape)  # torch.Size([100, 2048])
                # trip_pred = torch.sigmoid(self.mlp_pred(combined_fea))
                trip_pred = self.mlp_pred(combined_fea)
                clip_trip_pred_list.append(trip_pred)

            # clip_trip_pred_list = []
            # for i in range(num_frames):
            #     triplet_pred = torch.zeros(self.num_hois)
            #     for j in range(self.num_hois):
            #         action_name = self.HOIs[j].split(",")[1]
            #         act_p = clip_learned_action_prompt_list[i][self._actions.index(action_name)]
            #         hoi_p = vis_prompt_fea[i].squeeze(0)[j]
            #         triplet_pred[j] = torch.sigmoid(self.mlp_pred(torch.cat((act_p, hoi_p), dim=0)))
            #     clip_trip_pred_list.append(triplet_pred)
                # print("triplet_pred shape: ", triplet_pred.shape)  # torch.Size([100])
                # print("triplet_pred: ", triplet_pred)
            # print("clip_trip_pred_list: ",torch.stack(clip_trip_pred_list).shape)  # torch.Size([8, 100])
            batch_trip_pred.append(torch.stack(clip_trip_pred_list))

            # time6 = time.time()
            # if args.local_rank == 0:
            #     print("section 5 takes: ", time6 - time5)
            #     print("-" * 50)

            ##################################
            #        Action prediction       #
            ##################################
            clip_act_pred_list = []
            for i in range(num_frames):
                act_pred = self.mlp_pred_act(
                    clip_learned_action_prompt_list[i])
                clip_act_pred_list.append(act_pred)
            batch_act_pred.append(torch.stack(clip_act_pred_list))

        # batch_trip_pred_tensor = torch.stack(batch_trip_pred)
        # print("batch_trip_pred: ", len(batch_trip_pred))  # 40

        # torch.Size([40, 8, 100]) torch.Size([40, 8, 10])
        return torch.stack(batch_trip_pred), torch.stack(batch_act_pred), cap_prompt_fea, vis_prompt_fea, act_fea, dy_act_fea
        # return cap_prompt_fea, vis_prompt_fea
