"""
train + val
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torchvision import transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from timm.models.layers import trunc_normal_
from sklearn.metrics import average_precision_score
import wandb
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import os
import warnings
import tqdm
from tqdm import tqdm
import gc
import time
import random
import argparse
from datetime import datetime
from models.args import get_args
from data.CholecT50_dataloader2 import CholecT50, collate_fn_vid
from models.ACoLP_n1 import ACoLP
# import ivtmetrics

# torch.multiprocessing.set_sharing_strategy('file_system')

activation = nn.Sigmoid()
# wandb.login()

# %% evaluation metrics
# mAP = ivtmetrics.Recognition(100)
# # mAP.reset_global()
# mAPi = ivtmetrics.Recognition(6)
# mAPv = ivtmetrics.Recognition(10)
# mAPt = ivtmetrics.Recognition(15)
# # mAPi.reset_global()
# # mAPv.reset_global()
# # mAPt.reset_global()
# # print("Metrics built ...")


def main():
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '5678'
    args = get_args()
    # args.world_size = args.gpus * args.nodes
    # device = torch.device('cuda')
    device = torch.device(args.device)
    BATCH_SIZE = args.batch_size
    exp_time = datetime.now().strftime("%Y-%m-%d_%H:%M")

    if args.local_rank == 0:
        print("#" * 80)
        # print("# - Experiment: {}".format(exp_descp))
        print("# - Experiment start on: {}".format(exp_time))
        print("# - {}".format(args))
        print("#" * 80)

    # seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    gc.collect()  # empty RAM
    torch.cuda.empty_cache()

    # gpus=[0, 1, 2, 3]
    # dist_url = "env://"
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    ROOT_PATH = '/home/da/Desktop//CholecT50_complete/CholecT50/'
    _train_split = ['01', '15', '26', '40', '52', '65', '79', '02', '18', '27', '43', '56', '66',
                    '92', '04', '22', '31', '47', '57', '68', '96', '05', '23', '35', '48', '60',
                    '70', '103', '13', '25', '36', '49', '62', '75', '110']
    _train_split1 = ['70', '110']
    _val_split = ['08', '12', '29', '50', '78']
    _test_split = ['06', '51', '10', '73', '14', '74', '32', '80', '42', '111']
    experiment_id = datetime.now().strftime("%Y-%m-%d_%H_%M")
    save_path = os.path.join(ROOT_PATH, 'results/experiment_' + str(experiment_id))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        pass

    if not args.pre_save_img_fea:
        train_dst = CholecT50(
            root_dir=ROOT_PATH,
            mode='train',
            train_split=_train_split,
            val_split=_val_split,
            test_split=_test_split,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                ]
            )
        )

        val_dst = CholecT50(
            root_dir=ROOT_PATH,
            mode='val',
            train_split=_train_split,
            val_split=_val_split,
            test_split=_test_split,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                ]
            )
        )

        test_dst = CholecT50(
            root_dir=ROOT_PATH,
            mode='test',
            train_split=_train_split,
            val_split=_val_split,
            test_split=_test_split,
            transform=transforms.Compose(
                [
                    transforms.Resize((256, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                         0.229, 0.224, 0.225])
                ]
            )
        )
    else:
        train_dst = CholecT50(root_dir=ROOT_PATH,
                              train_split=_train_split, val_split=_val_split, test_split=_test_split, mode='train')
        train_dst1 = CholecT50(root_dir=ROOT_PATH,
                               train_split=_train_split1, val_split=_val_split, test_split=_test_split, mode='train1')
        val_dst = CholecT50(root_dir=ROOT_PATH,
                            train_split=_train_split, val_split=_val_split, test_split=_test_split, mode='val')
        test_dst = CholecT50(root_dir=ROOT_PATH,
                             train_split=_train_split, val_split=_val_split, test_split=_test_split, mode='test')

    if args.local_rank == 0:
        print("Train size:", len(train_dst))  # Train size: 27087
        print("Val size:", len(val_dst))  # Val size: 3216
        print("Test size:", len(test_dst))  # Val size: 321

    model = ACoLP()

    # from data.CholecT50_dataloder import collate_fn
    if args.distri:
        # model.apply(init_weights)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model.to(device), device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True)
        train_sampler = DistributedSampler(train_dst1, num_replicas=None, rank=None, shuffle=True,
                                           seed=42, drop_last=False)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size=BATCH_SIZE // 6,
                                                            drop_last=True)
        train_dataloader = DataLoader(train_dst1, num_workers=8, batch_sampler=train_batch_sampler,
                                      prefetch_factor=BATCH_SIZE, pin_memory=True,
                                      persistent_workers=True, collate_fn=collate_fn_vid)
        val_sampler = DistributedSampler(val_dst, num_replicas=None, rank=None, shuffle=False,
                                         seed=42, drop_last=False)
        val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size=BATCH_SIZE // 6,
                                                          drop_last=True)
        val_dataloader = DataLoader(val_dst, num_workers=8, batch_sampler=val_batch_sampler,
                                    prefetch_factor=BATCH_SIZE, pin_memory=True,
                                    persistent_workers=True, collate_fn=collate_fn_vid)
    else:
        # model.apply(init_weights).to(device)
        t = time.time()
        model.to(device)
        train_dataloader = DataLoader(train_dst, batch_size=BATCH_SIZE // 6, num_workers=8,
                                      shuffle=True, drop_last=False, prefetch_factor=BATCH_SIZE // 6,
                                      pin_memory=True, persistent_workers=True, collate_fn=collate_fn_vid)
        val_dataloader = DataLoader(val_dst, batch_size=BATCH_SIZE // 6, num_workers=8,
                                    shuffle=True, drop_last=False, prefetch_factor=BATCH_SIZE // 6,
                                    pin_memory=True, persistent_workers=True, collate_fn=collate_fn_vid)
        # len(train_dst) / batch_size = 1515
        print("length of dataloader: ", len(train_dataloader))
        dataload_time = time.time()
        print("loading data time; ", dataload_time - t)

    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr * args.lr_scale, weight_decay=args.weight_decay)
    # optimizer = optim.AdamW(model.parameters(), lr=args.lr * args.lr_scale)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr * args.lr_scale)
    exp_lr_scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.9)
    accumulate_step = 8

    # print("START TRAINING ...")
    # model.train()
    best_map_ivt = 0.0
    # run = wandb.init(
    #     # Set the project where this run will be logged
    #     project="Surgical_Triplet",
    #     # Track hyperparameters and run metadata
    #     config={
    #         "learning_rate": args.lr * args.lr_scale,
    #         "epochs": args.epochs,
    #     })

    for epoch in range(0, args.epochs):
        # mAP.reset()
        # start = time.time()
        # torch.cuda.empty_cache()
        total_step = len(train_dataloader)
        if args.local_rank == 0:
            print("Epoch:{}/{}".format(epoch, args.epochs-1))
            print("total steps: ", total_step)
        epoch_since = time.time()
        hs = open(os.path.join(save_path, "output.txt"), "a")

        if args.local_rank == 0:
            print("Learning Rate: ", str(exp_lr_scheduler.get_last_lr()))
        """
        batch 1:
            idx 1: ['VID08/000800.png', 'VID08/000801.png', ..., 'VID08/000807.png']
            idx 2: ['VID06/000800.png', 'VID06/000801.png', ..., 'VID06/000807.png']
            ...
            idx 40: []
        batch 2:
            ...
        """
        num_iter = 0.
        total_loss, trip_loss, verb_loss = 0.0, 0.0, 0.0
        total_loss_val, trip_loss_val, verb_loss_val = 0.0, 0.0, 0.0
        train_loss, train_loss_trip, train_loss_verb = [], [], []
        val_loss, val_loss_trip, val_loss_verb = [], [], []
        model.train()
        if args.local_rank == 0:
            print("START TRAINING ...")
        scaler = torch.cuda.amp.GradScaler()

        for k, sample_batched in enumerate(train_dataloader):
            start = time.time()
            # optimizer.zero_grad()
            # print("sample batched: ", len(sample_batched['frame names'][0]))  # 8
            # print("sample batched: ", len(sample_batched))  # 3
            # print("sample batched: ", sample_batched[0][1].shape)   # torch.Size([8, 3, 256, 448])
            # torch.Size([40, 8, 3, 256, 448])
            # print("sample batched: ", sample_batched[0].shape)
            # print("sample batched: ", sample_batched)   # torch.Size([40, 8, 3, 256, 448])
            num_iter += 1
            # inputs, trip_labels, frame_names = sample_batched[0].requires_grad_(True).to(device), \
            #                       sample_batched[1].to(device), \
            #                       sample_batched[2]
            trip_labels, verb_labels, frame_names = sample_batched[
                0], sample_batched[1], sample_batched[2]
            # if args.local_rank == 0:
            # print("inputs shape: ", inputs.shape)  # torch.Size([40, 8, 3, 256, 448])
            # torch.Size([40, 8, 100])
            # print("trip_labels shape: ", trip_labels.shape)
            # trip_pred = model.forward(inputs, frame_names)

            if not args.half_preci:
                trip_pred, act_pred = model.forward(frame_names)
                model_time = time.time()
                # if args.local_rank == 0:
                # print("model process time: ", model_time - start)
                # print("trip_pred: ", trip_pred[0])
                # torch.Size([40, 8, 100])
                # print("trip_pred: ", trip_pred.shape)
                # [tensor([]), tensor([]), ..., tensor([])] lenght = 8
                _trip_pred = trip_pred.view(-1, 100).type('torch.FloatTensor')
                # print(_trip_pred.shape) # torch.Size([72, 100])
                _act_pred = act_pred.view(-1, 10).type('torch.FloatTensor')
                _trip_labels = trip_labels.to(
                    device).view(-1, 100).type('torch.FloatTensor')
                _verb_labels = verb_labels.to(
                    device).view(-1, 10).type('torch.FloatTensor')
                loss_trip = criterion(_trip_pred, _trip_labels)
                loss_act = criterion(_act_pred, _verb_labels)
                loss = loss_trip + args.verb_loss_weight * loss_act
                # mAP.update(_trip_labels.detach().numpy(),
                #            activation(_trip_pred).detach().numpy())
                if args.accumulate:
                    print("accumulating gradients ...")
                    # accumulate_step = 8
                    loss = loss / accumulate_step
                    total_loss += loss.item()
                    loss.backward()
                    if (num_iter + 1) % accumulate_step == 0 or (num_iter + 1) % len(train_dataloader) == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                else:
                    optimizer.zero_grad(set_to_none=True)

                    # trip_loss += loss_trip.item()
                    # verb_loss += args.verb_loss_weight * loss_act.item()
                    # total_loss += loss.item()

                    train_loss_trip.append(loss_trip.item())
                    train_loss_verb.append(
                        args.verb_loss_weight * loss_act.item())
                    train_loss.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    trip_pred, act_pred = model.forward(frame_names)
                    # model_time = time.time()
                    # if args.local_rank == 0:
                    #     print("model process time: ", model_time - start)
                    #     # torch.Size([40, 8, 100])
                    #     print("trip_pred: ", trip_pred.shape)
                    #     # print("trip_pred: ", trip_pred[0])  # [tensor([]), tensor([]), ..., tensor([])] lenght = 8
                    _trip_pred = trip_pred.view(-1,
                                                100).type('torch.FloatTensor')
                    _act_pred = act_pred.view(-1, 10).type('torch.FloatTensor')
                    _trip_labels = trip_labels.view(-1,
                                                    100).type('torch.FloatTensor')
                    _verb_labels = verb_labels.to(
                        device).view(-1, 10).type('torch.FloatTensor')
                    loss_trip = criterion(_trip_pred, _trip_labels)
                    loss_act = criterion(_act_pred, _verb_labels)
                    loss = loss_trip + args.verb_loss_weight * loss_act
                    if args.accumulate:
                        accumulate_step = args.lr_scale
                        loss = loss / accumulate_step
                        total_loss += loss.item()
                        # loss.backward()
                        if (num_iter + 1) % accumulate_step == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                    else:
                        train_loss_trip.append(loss_trip.item())
                        train_loss_verb.append(
                            args.verb_loss_weight * loss_act.item())
                        train_loss.append(loss.item())

                        # # optimizer.zero_grad(set_to_none=True)
                        # total_loss += loss.item()
                        # # loss.backward()
                        # # optimizer.step()
                        optimizer.zero_grad()
                scaler.scale(loss).backward()
                # if (num_iter + 1) % 4 == 0:
                scaler.step(optimizer)
                scaler.update()

            # backward_time = time.time()
            # print("backwerd time: ", backward_time - model_time)
            dist.barrier()
            # if args.local_rank == 0:
            #     print("total loss: ", total_loss)
        exp_lr_scheduler.step()

        AP_list = []
        if epoch % args.val_interval == 0:
            # mAP.reset_global()
            # mAP.reset()
            if args.local_rank == 0:
                print("Validation @ epoch: ", epoch)
            with torch.no_grad():
                cunt = 0

                # AP_list = []
                for val_vid in _val_split:
                    trip_pred_list, trip_label_list = [], []
                    val_dst = CholecT50(root_dir=ROOT_PATH,
                                        train_split=_train_split, val_split=[val_vid], test_split=_test_split, mode='val')
                    val_sampler = DistributedSampler(val_dst, num_replicas=None, rank=None, shuffle=False,
                                                     seed=42, drop_last=False)
                    val_batch_sampler = torch.utils.data.BatchSampler(val_sampler, batch_size=BATCH_SIZE // 6,
                                                                      drop_last=True)
                    val_dataloader = DataLoader(val_dst, num_workers=8, batch_sampler=val_batch_sampler,
                                                prefetch_factor=BATCH_SIZE, pin_memory=True,
                                                persistent_workers=True, collate_fn=collate_fn_vid)
                    for _, val_sample in enumerate(val_dataloader):
                        cunt += 1
                        val_trip_labels, val_verb_labels, val_frame_names = val_sample[
                            0], val_sample[1], val_sample[2]
                        model.eval()
                        trip_pred, act_pred = model.forward(val_frame_names)
                        # print("act_pred shape: ", act_pred.shape)  # torch.Size([9, 8, 10, 1])
                        # print("trip_pred: ", trip_pred)
                        # print("trip_pred shape: ", trip_pred.shape) # torch.Size([9, 8, 100, 1])
                        # trip_mAP = mAP_update2(
                        #     activation(trip_pred).squeeze(-1).view(-1, 100).detach().cpu(), trip_labels.squeeze(-1).view(-1, 100).detach().cpu())[0]
                        # print("trip_mAP: ", trip_mAP)
                        _trip_pred = trip_pred.view(-1,
                                                    100).type('torch.FloatTensor')
                        # print(_trip_pred.shape) # torch.Size([72, 100])
                        _act_pred = act_pred.view(-1,
                                                  10).type('torch.FloatTensor')
                        _trip_labels = val_trip_labels.to(
                            device).view(-1, 100).type('torch.FloatTensor')
                        _verb_labels = val_verb_labels.to(
                            device).view(-1, 10).type('torch.FloatTensor')
                        val_ls_trip = criterion(_trip_pred, _trip_labels)
                        val_ls_act = criterion(_act_pred, _verb_labels)
                        val_ls = val_ls_trip + args.verb_loss_weight * val_ls_act
                        # trip_loss_val += val_ls_trip.item()
                        # verb_loss_val += val_ls_act.item()
                        # total_loss_val += val_ls.item()
                        val_loss_trip.append(val_ls_trip.item())
                        val_loss_verb.append(
                            args.verb_loss_weight * val_ls_act.item())
                        val_loss.append(val_ls.item())

                        trip_pred1 = activation(
                            trip_pred).squeeze(-1).view(-1, 100).detach().cpu()
                        print("logits: ", trip_pred1[10])
                        trip_pred_list.append(trip_pred1)
                        trip_label_list.append(_trip_labels)
                    trip_pred_list = torch.cat(trip_pred_list, dim=0)
                    trip_label_list = torch.cat(trip_label_list, dim=0)
                    warnings.filterwarnings('ignore')
                    AP = average_precision_score(
                        trip_label_list, trip_pred_list, average=None)
                    # if args.local_rank == 0:
                    #     print(AP)
                    AP[AP == -0] = np.nan
                    meanAP = np.nanmean(AP)
                    AP_list.append(meanAP)
                    
        if args.local_rank == 0:
            print("Per video triplet AP: ", AP_list)
            print("mean triplet AP: ", np.mean(AP_list))
        if np.mean(AP_list) >= best_map_ivt:
            best_map_ivt = np.mean(AP_list)
            if args.local_rank == 0:
                hs.write(
                    f'Epoch {epoch} Best Val IVT mAP: {best_map_ivt:.4f} \t')
            if dist.get_rank() == 0:
                torch.save(model.state_dict(), os.path.join(
                    save_path, "best_model" + ".pth"))
        else:
            if args.local_rank == 0:
                hs.write(
                    f'Epoch {epoch} Best Val IVT mAP: {best_map_ivt:.4f} \t')

        train_loss_avg = np.mean(train_loss)
        train_loss_trip_avg = np.mean(train_loss_trip)
        train_loss_verb_avg = np.mean(train_loss_verb)
        val_loss_avg = np.mean(val_loss)
        val_loss_trip_avg = np.mean(val_loss_trip)
        val_loss_verb_avg = np.mean(val_loss_verb)

        if args.local_rank == 0:
            hs.write(
                f'Epoch: {epoch} Train Loss: {train_loss_avg: .3f} \t Trip Loss: {train_loss_trip_avg: .3f} \t Action Loss: {train_loss_verb_avg: .3f} \t')
            hs.write(
                f'Epoch: {epoch} Val Loss: {val_loss_avg: .3f} \t Val Trip Loss: {val_loss_trip_avg: .3f} \t Val Action Loss: {val_loss_verb_avg: .3f} \n')
            # hs.write('-' * 50)
        # print("input_list: ", len(input_list))  # 8
        # print("triplet label_list: ", len(trip_label_list))  # 8
        # print("instrument label_list: ", len(ins_label_list))  # 8

        # if epoch >= 200:
        #     if dist.get_rank() == 0:
        #         checkpoint = {"model": model.state_dict(
        #         ), "optimizer": optimizer.state_dict()}
        #         torch.save(checkpoint, os.path.join(
        #             save_path, "model_epoch_" + str(epoch) + ".pth"))
        # wandb.log({"train loss": train_loss_avg, "val loss": val_loss_avg})


if __name__ == '__main__':
    main()
