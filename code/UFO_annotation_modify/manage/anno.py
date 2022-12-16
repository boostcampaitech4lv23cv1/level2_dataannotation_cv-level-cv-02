import os
import json
import numpy as np
from itertools import chain
from PIL import ImageOps, Image, ImageDraw
from typing import Sequence, Tuple


class ImageManager:
    """ImageManager
    image 경로를 입력으로 받아
    image를 다루는 class
    
    Args:
        filename(str) : the image path
        anno : the annotation file
    """
    def __init__(self, filename, anno):
        self._filename = filename
        self._img = Image.open(filename)
        self.point = Tuple[int, int]
        self.anno = anno

    def resize_img(self, img:Image, target_h: int = 1000):
        h, w = img.height, img.width
        ratio = target_h/h
        target_w = int(ratio * w)
        img = img.resize((target_w, target_h))
        return img, ratio
    
    def read_img(self, path: str, target_h: int = 1000) -> Image:
        """이미지 로드 후 텍스트 영역 폴리곤을 표시하여 반환한다."""
        # load image, annotation
        img = Image.open(path)
        #img = ImageOps.exif_transpose(img)  # 이미지 정보에 따라 이미지를 회전
        ann = self.anno
        words = ann['words']

        # resize
        img, ratio = self.resize_img(img)

        # draw polygon
        for key, val in words.items():
            poly = val['points']
            poly_resize = [[v * ratio for v in pt] for pt in poly]
            illegibility = val['illegibility']
            self.draw_polygon(img, poly_resize, illegibility)

        return img

    def draw_polygon(self, img: Image, pts, illegibility: bool):
        """이미지에 폴리곤을 그린다. illegibility의 여부에 따라 라인 색상이 다르다."""
        pts = list(chain(*pts)) + pts[0]  # flatten 후 첫번째 점을 마지막에 붙인다.
        img_draw = ImageDraw.Draw(img)
        # 폴리곤 선 너비 지정이 안되어 line으로 표시
        img_draw.line(pts, width=3, fill=(0, 255, 255) if not illegibility else (255, 0, 255))
    
    def crop_img(self, path: str, points: list, max_size:int = 400) -> Image:
        """이미지의 points 영역을 잘라낸다."""
        img = Image.open(path)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        img = img.crop((x_min,y_min, x_max, y_max))
        img, _ = self.resize_img(img, target_h=300)
        return img
    
class ReadManager:
    """ReadJson
    annotation 확인할 json 파일을 선택해 읽어오며
    수정사항이 있을시 수정한다.
    
    Args:
        filename(str) : the json file
    """
    
    def __init__(self, filename, root_path):
        self.root_path = root_path
        self.filename = filename
        self.data = dict()
        self.files = []
    
    def get_annotation_file(self):
        with open(os.path.join(self.root_path, self.filename)) as f:
            self.data = json.load(f)
        return self.data
    
    def set_annotation_files(self, files):
        self._annotations_files = files
    
    def get_image_files(self):
        self.files = list(self.data['images'].keys())
        return self.files
        
    def get_image(self, index):
        return self.files[index]

    def update_data(self, filename, anno_idx, update_anno):
        # update_anno = update_anno.encode('ascii', 'backslashreplace').decode('utf-8')
        print(filename, anno_idx, update_anno)
        print(self.data['images'][filename])
        self.data['images'][filename]['words'][str(anno_idx)].update({'transcription' : update_anno})
    
    def save_data(self):
        with open(os.path.join(self.root_path, self.filename), 'w') as f:
            json.dump(self.data, f, indent=4)