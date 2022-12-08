import wandb 
import numpy as np 
import matplotlib.pyplot as plt 
import json
import cv2
import os.path as osp
from detect import detect 
import re

IMG_PATH = "/opt/ml/input/data/ICDAR17_Korean/images"
VAL_DATA_PATH = "/opt/ml/input/data/ICDAR17_Korean/ufo/random_split/val.json"
INFERENCE_SHAPE = 1024

with open(VAL_DATA_PATH, "rb") as f: 
    val_json = json.load(f)
VAL_ID = list(val_json["images"].keys())
VAL_ID = sorted(VAL_ID, key=lambda f: int(re.sub('\D', '', f)))

def make_wandb_table(model, loss):
    table =  wandb.Table(columns = ["fname", "image", "loss"])
    for _, loss, val_idx  in loss:
        file_name = VAL_ID[val_idx]

        #LIST가 중요함.
        img = [cv2.imread(osp.join(IMG_PATH, file_name))[:, :, ::-1]]

        fig, ax = plt.subplots(1,1)
        model.eval()
        prediction = detect(model, img, INFERENCE_SHAPE)[0]
        ground_truth = val_json["images"][file_name]["words"]
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
        table.add_data(file_name, wandb.Image(fig),loss)
    return table