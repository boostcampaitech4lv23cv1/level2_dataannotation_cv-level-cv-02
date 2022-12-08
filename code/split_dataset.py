import json 
from sklearn.model_selection import train_test_split
import random 
from pathlib import Path

DATASET_PATH = "../input/data/ICDAR17_Korean/ufo/train.json"

with open("../input/data/ICDAR17_Korean/ufo/train.json", "rb") as f: 
    train_json = json.load(f)

print("train_json", len(list(train_json["images"].keys())))


#(TODO) 빠르게 random_split해서 json 파일 만들기

def random_split_dataset(original_json, output_path, random_seed = 2022):
    id_pair = list(original_json["images"].keys())
    random.seed(random_seed)
    random.shuffle(id_pair)

    print("LENGTH:", len(id_pair))
    train_cnt = int(len(id_pair) * 0.8)
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

OUTPUT_PATH = "../input/data/ICDAR17_Korean/ufo/random_split"

random_split_dataset(train_json, OUTPUT_PATH)

