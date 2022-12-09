import json 
from sklearn.model_selection import train_test_split
import random 
from pathlib import Path
import os 
import numpy as np 
import torch

from argparse import ArgumentParser

from base import TOKEN_TO_PATH 


def seed_everything(seed: int = 2022):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument('--data_name', type=str, default="nothing")
    parser.add_argument('--split_ratio', type=float, default="0.8")   #split ratio
    
    args = parser.parse_args()
    return args

def get_full_json(path):
    with open(path, "rb") as f: 
        train_json = json.load(f)
    return train_json


#(TODO) 빠르게 random_split해서 json 파일 만들기

def random_split_dataset(original_json, output_path, split_ratio, random_seed = 2022):
    id_pair = list(original_json["images"].keys())
    random.Random(SEED).shuffle(id_pair)
    print("LENGTH:", len(id_pair))
    train_cnt = int(len(id_pair) * split_ratio)
    print("TRAIN:", train_cnt)
    tr_id = set(id_pair[:train_cnt])
    val_id = set(id_pair[train_cnt:])

    assert len(tr_id.intersection(val_id))==0, "Duplicate id"
    train_info = dict() 
    val_info = dict()

    for img_id, value in original_json["images"].items():
        if img_id in tr_id:
            train_info[img_id] = value        
        else:
            val_info[img_id] = value
    
    train_json = dict(images = train_info)
    val_json = dict(images = val_info)

    Path(output_path).mkdir(exist_ok = True)

    train_path = Path(output_path) / "train.json"
    val_path = Path(output_path) / "val.json"

    with open(train_path, "w") as f:
        json.dump(train_json, f)
    with open(val_path, "w") as f:
        json.dump(val_json, f)
    
    print("FINISHED")
    return


SEED = 2022
seed_everything(SEED)
args = parse_args()

data_name = TOKEN_TO_PATH.get(args.data_name, "nothing")
assert data_name != "nothing", f"변환할 데이터셋의 토큰을 정확히 지정해주세요. 가능한 후보 : {list(TOKEN_TO_PATH.keys())} , 입력하신 것: {data_name}입니다."

DATASET_PATH = f"../input/data/{data_name}/ufo/train.json"
OUTPUT_PATH = f"../input/data/{data_name}/ufo/random_split"


def main():
    print(f"선택하신 dataset은 {data_name}입니다.")
    data = get_full_json(DATASET_PATH)
    random_split_dataset(data, OUTPUT_PATH, SEED)

if __name__ == "__main__":
    main()

