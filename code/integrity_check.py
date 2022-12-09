import json 
import os

from base import TOKEN_TO_PATH
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    # Conventional args
    parser.add_argument('--data_name', type=str,
                        default="nothing")
    parser.add_argument('--mode', type=str,
                        default="debug")
    
    args = parser.parse_args()
    return args

args = parse_args()

data_name = TOKEN_TO_PATH.get(args.data_name, "nothing")
assert data_name != "nothing", f"변환할 데이터셋의 토큰을 정확히 지정해주세요. 가능한 후보 : {list(TOKEN_TO_PATH.keys())} , 입력하신 것: {data_name}입니다."

JSON_PATH = f"/opt/ml/input/data/{data_name}/ufo/train.json"
FIX_DATA = args.mode == "fix"


print("Check dataset type:", data_name)
print("Fix your data????:", FIX_DATA)

def get_empty_data(json_path):
    empty_lists = []
    with open(json_path, "rb") as f: 
        full_data = json.load(f)
    for key, val in full_data["images"].items():
        word_info = val["words"]
        if len(word_info) == 0 :
            empty_lists.append(key)
    empty_set = set(empty_lists)      
    return empty_set

def get_excepted_data(json_path):
    excepted_lists = []
    with open(json_path, "rb") as f: 
        full_data = json.load(f)
    for key, val in full_data["images"].items():
        word_info = val["words"]
        area_cnt = 0
        for word_cnt, word_val in word_info.items():
            if word_val["illegibility"] : 
                continue 
            else: 
                area_cnt +=1
        if area_cnt == 0 :
            excepted_lists.append(key)
    excepted_set = set(excepted_lists)      
    return excepted_set


def make_new_json(json_path):
    empty_set = get_empty_data(json_path)
    with open(json_path, "rb") as f: 
        full_data = json.load(f)

    new_dict = {}
    for key, val in full_data["images"].items():
        if key in empty_set: continue
        new_dict[key] = val
    result_json = dict(images = new_dict)

    with open(json_path, "w") as f:
        json.dump(result_json, f)

def main():
    empty_set = get_empty_data(JSON_PATH)
    meaningless_set = get_excepted_data(JSON_PATH)

    if empty_set or meaningless_set:
        print("Annotation이 없는 empty data:", empty_set)
        print("제외영역밖에 없는 data:", meaningless_set)
        print("잘못된 데이터가 발견되었습니다. 필요에 따라 make_new_json 함수를 이용하여 train.json을 새롭게 만들어주세요.")
        if FIX_DATA:
            make_new_json(JSON_PATH)

if __name__ == "__main__":
    main()