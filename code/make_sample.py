import os 
import json 
import os.path as osp 
import shutil
from pathlib import Path
import pickle
 

#지금 내가 작업한건 윈도우인데, 너는 맥북이니까
# /input/data/aihub_final_easy 이런 식으로 작업하면 될거야

COMMON_PATH = osp.join("input", "data", "aihub_final_easy")
COPY_DIRNAME = "aihub_real_final"
COPY_PATH = osp.join("input", "data", COPY_DIRNAME)
THRESHOLD = 1500

JSON_PATH = osp.join(COMMON_PATH, "ufo", "train.json")
IMG_PATH = osp.join(COMMON_PATH, "images")

with open(JSON_PATH, "rb") as f: 
    train_json = json.load(f)

SRC_IMG_PATH = IMG_PATH
DEST_IMG_PATH = osp.join(COPY_PATH, "images")

SRC_JSON_PATH = JSON_PATH
DEST_JSON_PATH = osp.join(COPY_PATH, "ufo", "train.json")

def make_new_json(src_json_path, dest_json_path, src_set, exclude = False):
    with open(src_json_path, "rb") as f: 
        full_data = json.load(f)

    new_dict = {}
    if not exclude:
        for key, val in full_data["images"].items():
            if key in src_set: 
                print("key is here:", key, "exclude:", exclude)
                new_dict[key] = val
    else:
        for key, val in full_data["images"].items():
            if key not in src_set: 
                print("key is here:", key, "exclude:", exclude)
                new_dict[key] = val
    result_json = dict(images = new_dict)

    with open(dest_json_path, "w") as f:
        json.dump(result_json, f)

def make_new_image(src_img_path, dest_img_path, src_set, exclude = False):

    if not exclude:
        for idx, fname in enumerate(os.listdir(src_img_path)):
            name, ext = osp.splitext(fname)
            order = int(name.split("_")[-1])

            src_file_path = osp.join(src_img_path, fname)
            dest_file_path = osp.join(dest_img_path, fname)

            if fname in src_set:
                shutil.copy(src_file_path, dest_file_path)
    else:
        for idx, fname in enumerate(os.listdir(src_img_path)):
            name, ext = osp.splitext(fname)
            order = int(name.split("_")[-1])

            src_file_path = osp.join(src_img_path, fname)
            dest_file_path = osp.join(dest_img_path, fname)
            if fname not in src_set:
                shutil.copy(src_file_path, dest_file_path)

## 수작업으로 할거라면 ignore_lists를 별도로 가공하면 충분함

with open("final_ignore_list.pickle", "rb") as f:
    ignore_lists = pickle.load(f)

print("ignore_lists:", ignore_lists[:10])
print("length:", len(ignore_lists))

print("SRC:", SRC_IMG_PATH, "JSON:", SRC_JSON_PATH)
print("DEST:", DEST_IMG_PATH, "DEST JSON:", DEST_JSON_PATH)


# make_new_json(SRC_JSON_PATH, DEST_JSON_PATH , ignore_lists, exclude = True)
# make_new_image(SRC_IMG_PATH, DEST_IMG_PATH , ignore_lists, exclude = True)
