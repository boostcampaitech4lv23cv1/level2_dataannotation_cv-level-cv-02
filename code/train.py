import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
from detect import detect

import wandb
import matplotlib.pyplot as plt
import numpy as np
import json

import random
import re
import cv2
from torch.utils.data import ConcatDataset
from base import TOKEN_TO_PATH, DATASETS_TO_USE
from utils import make_wandb_table, CosineAnnealingWarmUpRestarts

INFERENCE_SHAPE = 1024

#####################################
# (TODO) 학습시킬 PATH를 별도로 저장.
# (TODO) data가 저장되어있는 곳들을 이렇게 담아주세요


# Key   : Simple acronym for dataset
# Value : full name for data directory path. Change it according to your path
# e.g.) if your ICDAR19_Korean dataset is downloaded as "ic19",
#       change "ICDAR19_Korean" ==> "ic19"
# But we strongly recommend to download the dataset with shell script files
###################################3

"""
TOKEN_TO_PATH : key(축약어) - value(실제 데이터셋경로)로 지정된 dictionary
DATASETS_TO_USE : 여러 dataset 중에서 어떤 것을 사용할지를 담은 list

Example) 
>>> TOKEN_TO_PATH 
>>> {
    "ko17" : "ICDAR17_Korean" ,
    "ko19" : "ICDAR19_Korean" ,
    "camper" : "boostcamp" , 
    "aihub"  : "aihub"}

>>> DATASETS_TO_USE 
>>> ["ko17", "camper"]


또한, 각 데이터셋에 대해 다음과 같은 가정을 상정한다.

ex) {데이터셋명} = ICDAR17_Korean  (베이스라인 코드에 이렇게 저장되어있음)
[1] 데이터셋의 경로는 /opt/ml/input/data/{데이터셋명}"
[2] 이미지의 경로는 /opt/ml/input/data/{데이터셋명}/images"
[3] VAL_JSON의 경로는 /opt/ml/input/data/{데이터셋명}/ufo/random_split/val.json"

"""

PREDICTION_RESIZE = 512

def setup_wandb(run_name = "Custom_run_name"):
    wandb.init(
        project = "DataAnnotation",
        config = args
    )

    wandb.run_name = run_name

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', '../input/data/ICDAR17_Korean'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--input_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)

    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--save_interval', type=int, default=1)

    parser.add_argument('--start_early_stopping', type=int, default=20)   ## early stopping count 시작 epoch
    parser.add_argument('--early_stopping_patience', type=int, default=5)   ## early stopping patience

    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir,
                device, image_size,
                input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval,
                start_early_stopping, early_stopping_patience):

    setup_wandb()

    #train_data_stack, val_data_stack에 dataset을 stack한다.
    train_data_stack = []
    val_data_stack = []

    for dataset_type in DATASETS_TO_USE:
        data_name = TOKEN_TO_PATH.get(dataset_type, "invalid")
        assert data_name != "invalid" , "잘못된 데이터셋 이름을 입력하셨습니다. 다운로드한 데이터셋 환경을 확인해주세요"

        DATA_PATH = f"/opt/ml/input/data/{data_name}"
        IMG_PATH = f"/opt/ml/input/data/{data_name}/images"
        VAL_DATA_PATH = f"/opt/ml/input/data/{data_name}/ufo/random_split/val.json"

        sub_train_dataset = SceneTextDataset(DATA_PATH, split='train', image_size=image_size, crop_size=input_size)
        sub_train_dataset = EASTDataset(sub_train_dataset)
        train_data_stack.append(sub_train_dataset)

        sub_val_dataset = SceneTextDataset(DATA_PATH, split='val', image_size=image_size, crop_size=input_size)
        sub_val_dataset = EASTDataset(sub_val_dataset)
        val_data_stack.append(sub_val_dataset)

    # ConcatDataset을 통해 다 합쳐준다.
    train_dataset = ConcatDataset(train_data_stack)
    val_dataset = ConcatDataset(val_data_stack)

    train_num_batches = math.ceil(len(train_dataset) / batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    #(TODO) train_dataset을 val_dataset으로 대체

    val_num_batches = math.ceil(len(val_dataset) / batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EAST()
    model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr= 1e-4)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 10, T_mult= 2 , eta_max =0.05, T_up = 2, gamma = 0.5)

    current_lr = scheduler.get_lr()[0]


    model.train()

    for epoch in range(max_epoch):
        model.train()
        train_epoch_loss, epoch_start = 0, time.time()
        with tqdm(total=train_num_batches) as pbar:
            for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                pbar.set_description('[Epoch {}]'.format(epoch + 1))

                if torch.sum(gt_score_map) < 1 : 
                    # print("Skip this label")
                    pbar.update(1)
                    continue

                loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_val = loss.item()
                train_epoch_loss += loss_val
                pbar.update(1)

                val_dict = {
                        'Train Cls loss': extra_info['cls_loss'], 'Train Angle loss': extra_info['angle_loss'],
                        'Train IoU loss': extra_info['iou_loss'] , "learning_rate" : current_lr
                    }
                wandb.log(val_dict)
                pbar.set_postfix(val_dict)

        scheduler.step()

        print('Training Mean loss: {:.4f} | Elapsed time: {}'.format(
            train_epoch_loss / train_num_batches, timedelta(seconds=time.time() - epoch_start)))
       
        wandb.log({"Train_loss" : train_epoch_loss / train_num_batches})

        #Model을 validation으로 바꿔줌
        model.eval()

        with tqdm(total=val_num_batches) as pbar:
            val_epoch_loss, val_epoch_start = 0, time.time()

            with torch.no_grad():
                for val_idx , (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(val_loader):
                    pbar.set_description('[Epoch {} Validation]'.format(epoch + 1))

                    if torch.sum(gt_score_map) < 1 : 
                        # print("Skip this label")
                        pbar.update(1)
                        continue

                    loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                    loss_val = loss.item()
                    val_epoch_loss += loss_val
                    pbar.update(1)
                    val_dict = {
                        'Val Cls loss': extra_info['cls_loss'], 'Val Angle loss': extra_info['angle_loss'],
                        'Val IoU loss': extra_info['iou_loss']
                    }

                    wandb.log(val_dict)
                    pbar.set_postfix(val_dict)

        val_loss = val_epoch_loss / val_num_batches

        print('Validation Mean loss: {:.4f} | Elapsed time: {}'.format(
            val_loss, timedelta(seconds=time.time() - val_epoch_start)))

        wandb.log({"Val_loss" : val_loss})
        # save_interval마다, 상위 10개에 대해 Loss sample을 분석!

        if (epoch +1) % save_interval == 0 : 
            EVAL_BATCH_SIZE = 1
            eval_num_batches = len(val_dataset)

            eval_loader = DataLoader(val_dataset, batch_size= EVAL_BATCH_SIZE , shuffle=False, num_workers=4)
            model.eval()
            eval_losses = []
            with tqdm(total= eval_num_batches) as pbar:
                with torch.no_grad():
                    for val_idx , (img, gt_score_map, gt_geo_map, roi_mask) in enumerate(eval_loader):
                        pbar.set_description('[Epoch {} EValuate]'.format(epoch + 1))
                        #(TODO) 사실은 val_step이랑 동일하다!

                        if torch.sum(gt_score_map) < 1 : 
                            # print("Skip this label")
                            pbar.update(1)
                            continue

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        loss_val = loss.item()
                        pbar.update(1)
                        eval_dict = {
                            'EVal Cls loss': extra_info['cls_loss'], 'EVal Angle loss': extra_info['angle_loss'],
                            'EVal IoU loss': extra_info['iou_loss']
                        }

                        eval_losses.append((loss_val, val_idx)) 
                        pbar.set_postfix(eval_dict)

            random_losses = random.sample(eval_losses, 10)
            eval_losses = sorted(eval_losses, key = lambda x: -x[0])[:10]
            top_loss_table = make_wandb_table(model, eval_losses)
            random_loss_table = make_wandb_table(model, random_losses)

            wandb.log({"TOP Loss10" : top_loss_table})
            wandb.log({"Random Loss" : random_loss_table})
            
        #끝났으니 다시 원래대로
        model.train()

        #save last.pth
        if (epoch + 1) % save_interval == 0:
            if not osp.exists(model_dir):
                os.makedirs(model_dir)

            ckpt_fpath = osp.join(model_dir, f'camper_icdar17_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_fpath)
        
        #save first loss
        if epoch == 0 : 
            best_loss = val_loss 

        #save best.pth
        if epoch >= start_early_stopping :
            if val_loss < best_loss :
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'best.pth')
                torch.save(model.state_dict(), ckpt_fpath)
                print("----- New best model in {}epoch -----".format(epoch+1))

                #save best loss
                best_loss = val_loss

                #initial count
                stopping_count = 0

            else : 
                stopping_count += 1

                #early stopping
                if stopping_count == early_stopping_patience :
                    print("----- Stop train in {}epoch -----".format(epoch+1))
                    break


def main(args):
    do_training(**args.__dict__)


if __name__ == '__main__':
    args = parse_args()
    print("args:", args)
    main(args)
