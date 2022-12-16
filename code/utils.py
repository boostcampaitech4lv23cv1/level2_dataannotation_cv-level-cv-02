import wandb 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import cv2
import os.path as osp
from detect import detect 
import re
from typing import List, Tuple, Dict
import PIL

from base import TOKEN_TO_PATH, DATASETS_TO_USE
from torch.optim.lr_scheduler import _LRScheduler
import math


print("DATASETS_TO_USE:", DATASETS_TO_USE)
print("TOKEN_TO_PATH:", TOKEN_TO_PATH)

INFERENCE_SHAPE = 1024



class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr




def get_json_path(path):
    """
    return sorted validation ID from annotation file
    """
    with open(path, "rb") as f: 
        val_json = json.load(f)
    VAL_ID = list(val_json["images"].keys())
    VAL_ID = sorted(VAL_ID, key=lambda f: int(re.sub('\D', '', f)))
    return VAL_ID

def get_val_id(dataset_types : List) -> Tuple[List, List]:
    """
    get validation ID from all annotations
    """
    val_id = []
    sources = []  #어느 dataset(directory)의 파일인지를 명시해줌
    for data_acronym in dataset_types:
        dataset_name = TOKEN_TO_PATH[data_acronym]
        val_path = f"/opt/ml/input/data/{dataset_name}/ufo/random_split/val.json"
        each_val_id = get_json_path(val_path)
        each_source = [dataset_name for _ in range(len(each_val_id))]
        val_id.extend(each_val_id)
        sources.extend(each_source)
    
    return val_id, sources

VAL_ID, SOURCES = get_val_id(DATASETS_TO_USE)

def make_wandb_table(model, losses):
    table =  wandb.Table(columns = ["source", "fname", "image", "loss"])
    for loss,  val_idx  in losses:
        file_name = VAL_ID[val_idx]
        source = SOURCES[val_idx]
        SOURCE_IMG_PATH = f"/opt/ml/input/data/{source}/images"
        SOURCE_JSON_PATH = f"/opt/ml/input/data/{source}/ufo/random_split/val.json"

        with open(SOURCE_JSON_PATH, "rb") as f:
            source_val_json = json.load(f)

        #LIST가 중요함.
        #img = [cv2.imread(osp.join(SOURCE_IMG_PATH, file_name))[:, :, ::-1]]
        img = [np.asarray(PIL.Image.open(osp.join(SOURCE_IMG_PATH, file_name)))]

        fig, ax = plt.subplots(1,1)
        model.eval()
        prediction = detect(model, img, INFERENCE_SHAPE)[0]

        ground_truth = source_val_json["images"][file_name]["words"]
        for idx, word in enumerate(prediction):
            word =word[::-1]
            word = np.append(word, word[0]).reshape(-1,2)
            for prev_pos, next_pos in zip(word[:-1], word[1:]):
                ax.plot( [prev_pos[0], next_pos[0]], [prev_pos[1], next_pos[1]]
                        ,color='b', linestyle='-', linewidth=1.5)
        
        for word_key, word_val in ground_truth.items():
            word = np.array(word_val["points"])
            word =word[::-1]
            word = np.append(word, word[0]).reshape(-1,2)
            for prev_pos, next_pos in zip(word[:-1], word[1:]):
                ax.plot( [prev_pos[0], next_pos[0]], [prev_pos[1], next_pos[1]]
                        ,color='r', linestyle='-', linewidth=1.5)
        ax.axis("off")
        ax.imshow(img[0])
        # top_loss_table.add_data(fig,loss)
        table.add_data(source, file_name, wandb.Image(fig), loss)
    return table