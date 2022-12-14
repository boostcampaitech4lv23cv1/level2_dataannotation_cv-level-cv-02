import json
import os
import os.path as osp
from glob import glob
from PIL import Image

import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, ConcatDataset, Dataset
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    # Conventional args
    parser.add_argument('--only_korean', type=str,
                        default="all")
    args = parser.parse_args()
    return args


args = parse_args()
CONVERT_KOREAN = False if args.only_korean == "all" else True

SRC_DATASET_DIR = '/opt/ml/input/data/ICDAR19'  # FIXME
DST_DATASET_DIR = f"{SRC_DATASET_DIR}_Korean" if CONVERT_KOREAN else SRC_DATASET_DIR

print("CONVERT_TO_KOREAN??:", CONVERT_KOREAN)
print("Save your file to :", DST_DATASET_DIR)

NUM_WORKERS = 4  # FIXME

IMAGE_EXTENSIONS = {'.gif', '.jpg', '.png'}

LANGUAGE_MAP = {
    'Korean': 'ko',
    'Latin': 'en',
    'Mixed': 'pass',
    'Symbols': None
}

def get_language_token(x):
    return LANGUAGE_MAP.get(x, 'others')

def maybe_mkdir(x):
    if not osp.exists(x):
        os.makedirs(x)

class MLT19Dataset(Dataset):
    def __init__(self, image_dir, label_dir, copy_images_to=None):
        image_paths = {x for x in glob(osp.join(image_dir, '*')) if osp.splitext(x)[1] in
                       IMAGE_EXTENSIONS}
        label_paths = set(glob(osp.join(label_dir, '*.txt')))
        assert len(image_paths) == len(label_paths)

        sample_ids, samples_info = list(), dict()
        for image_path in image_paths:
            sample_id = osp.splitext(osp.basename(image_path))[0]

            label_path = osp.join(label_dir, '{}.txt'.format(sample_id))
            assert label_path in label_paths

            words_info, extra_info = self.parse_label_file(label_path)

            if CONVERT_KOREAN:
                if 'ko' not in extra_info['languages'] or extra_info['languages'].difference({'ko', 'en'}):
                    continue

            sample_ids.append(sample_id)
            samples_info[sample_id] = dict(image_path=image_path, label_path=label_path,
                                           words_info=words_info)

        self.sample_ids, self.samples_info = sample_ids, samples_info

        self.copy_images_to = copy_images_to

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_info = self.samples_info[self.sample_ids[idx]]

        image_fname = osp.basename(sample_info['image_path'])
        image = Image.open(sample_info['image_path'])
        img_w, img_h = image.size

        if self.copy_images_to:
            maybe_mkdir(self.copy_images_to)
            image.save(osp.join(self.copy_images_to, osp.basename(sample_info['image_path'])))

        license_tag = dict(usability=True, public=True, commercial=True, type='CC-BY-SA',
                           holder=None)
        sample_info_ufo = dict(img_h=img_h, img_w=img_w, words=sample_info['words_info'], tags=None,
                               license_tag=license_tag)

        return image_fname, sample_info_ufo

    def parse_label_file(self, label_path):
        def rearrange_points(points):
            start_idx = np.argmin([np.linalg.norm(p, ord=1) for p in points])
            if start_idx != 0:
                points = np.roll(points, -start_idx, axis=0).tolist()
            return points

        with open(label_path, encoding='utf-8') as f:
            lines = f.readlines()

        words_info, languages = dict(), set()
        for word_idx, line in enumerate(lines):
            items = line.strip().split(',', 9)
            language, transcription = items[8], items[9]
            points = np.array(items[:8], dtype=np.float32).reshape(4, 2).tolist()
            points = rearrange_points(points)

            illegibility = transcription == '###'
            orientation = 'Horizontal'
            language = get_language_token(language)
            words_info[word_idx] = dict(
                points=points, transcription=transcription, language=[language],
                illegibility=illegibility, orientation=orientation, word_tags=None
            )
            languages.add(language)

        return words_info, dict(languages=languages)


def main():
    dst_image_dir = osp.join(DST_DATASET_DIR, 'images')
    # dst_image_dir = None

    mlt_train = MLT19Dataset(osp.join(SRC_DATASET_DIR, 'raw_images'),
                             osp.join(SRC_DATASET_DIR, 'gt'),
                             copy_images_to=dst_image_dir)

    anno = dict(images=dict())

    with tqdm(total=len(mlt_train)) as pbar:
        for batch in DataLoader(mlt_train, num_workers=NUM_WORKERS, collate_fn=lambda x: x):
            image_fname, sample_info = batch[0]
            anno['images'][image_fname] = sample_info
            pbar.update(1)

    ufo_dir = osp.join(DST_DATASET_DIR, 'ufo')
    maybe_mkdir(ufo_dir)
    with open(osp.join(ufo_dir, 'train.json'), 'w') as f:
        json.dump(anno, f, indent=4)


if __name__ == '__main__':
    main()
