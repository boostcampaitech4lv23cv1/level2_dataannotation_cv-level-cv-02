import os.path as osp
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

import numpy.random as npr
import albumentations as A
from albumentations.pytorch import ToTensorV2
from shapely.geometry import Polygon
# from shapely.validation import make_valid

"""
이 코드의 핵심로직은 다음 세 가지 함수 & 클래스.

[1] transform_by_matrix : 이미지에 matrix를 곱하여 변환한다.(모든 기하학적 변환은 행렬로 간주될 수 있다.)

[2] GeoTransformation은 이 행렬을 얻어내기 위한 로직이 담긴 클래스이다.
이 클래스를 부르면 crop_rotate_resize를 호출한다.

crop_rotate_resize 함수는, input에서 랜덤한 패치를 얻어내는 과정이다.

translate가 없다면 그냥 직사각형 patch, 아닐 경우 quadrilateral patch가 된다. (quad patch를 얻어내기 위한 함수는 get_located_patch_quad)
이렇게 얻어낸 patch ==> output space image box 사이의 변환관계 행렬을 얻어낸다. 단순히 말해서, Input space X(이미지의 일부) ==> Output space Y라면 Y = AX가 되는 A를 찾아내는 것.

[3] 이렇게 얻어낸 변환행렬 A를 이용하여, 본래의 이미지, bbox에 transform_by_matrix을 곱해 변환된 결과를 반환한다.

[4] 그 외의 다양한 클래스 변수(ex. self.crop_ratio...) 는, 결국 output patch의 좌표를 구하기 위한 로직들

- ex) 가로:세로 = 16:9로 지정, crop_size는 300으로 지정한다면 output_height, output_width는 16:9 = X : 300으로 이용하거나 16:9 = 300 : X로 구하면 되니깐.

그렇게 output patch를 구하는 것.

"""

def transform_by_matrix(matrix, image=None, oh=None, ow=None, word_bboxes=[],
                        by_word_char_bboxes=[], masks=[], inverse=False):
    """
    Args:
        matrix (ndarray): (3, 3) shaped transformation matrix.
        image (ndarray): (H, W, C) shaped ndarray.
        oh (int): Output height.
        ow (int): Output width.
        word_bboxes (List[ndarray]): List of (N, 2) shaped ndarrays.
        by_word_char_bboxes (List[ndarray]): Lists of (N, 4, 2) shaped ndarrays.
        masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        inverse (bool): Whether to apply inverse transformation.
    """

    # image, masks 명시했다면, oh와 ow가 반드시 명시되어야 함

    if image is not None or masks is not None:
        assert oh is not None and ow is not None

    output_dict = dict()


    #역행렬용 inverse를 구한다.

    if inverse:
        matrix = np.linalg.pinv(matrix)


    # matrix를 곱하여 원근변환을 준다. 일종의 shear.

    if image is not None:
        output_dict['image'] = cv2.warpPerspective(image, matrix, dsize=(ow, oh))

    if word_bboxes is None:
        output_dict['word_bboxes'] = None

    # (TODO) 로직 이해하기

    elif len(word_bboxes) > 0:
        num_points = list(map(len, word_bboxes))
        points = np.concatenate([np.reshape(bbox, (-1, 2)) for bbox in word_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 2)  # (N, 2)
        output_dict['word_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + num_points)[:-1], num_points)]
    else:
        output_dict['word_bboxes'] = []

    if by_word_char_bboxes is None:
        output_dict['by_word_char_bboxes'] = None
    elif len(by_word_char_bboxes) > 0:
        word_lens = list(map(len, by_word_char_bboxes))
        points = np.concatenate([np.reshape(bboxes, (-1, 2)) for bboxes in by_word_char_bboxes])  # (N, 2)
        points = cv2.perspectiveTransform(
            np.reshape(points, (1, -1, 2)).astype(np.float32), matrix).reshape(-1, 4, 2)  # (N, 4, 2)
        output_dict['by_word_char_bboxes'] = [
            points[i:i + n] for i, n in zip(np.cumsum([0] + word_lens)[:-1], word_lens)]
    else:
        output_dict['by_word_char_bboxes'] = []


    # mask를 명시해줬다면, mask에 matrix를 곱한 결과를 return해준다.

    if masks is None:
        output_dict['masks'] = None
    else:
        output_dict['masks'] = [cv2.warpPerspective(mask, matrix, dsize=(ow, oh)) for mask in masks]

    return output_dict


class GeoTransformation:
    """
    - 회전 관련 변환 :: rotate_anchors( -90, 90), rotate_range(-10,10)이라면... -90도나 90도 중 하나를 택한 후 그 각도 중 -10 ~ 10을 랜덤하게 더함.

    - crop 관련 변환 :: crop_aspect_ratio, crop_size, crop_size_by

    - flip 관련 변환 :: hflip, vflip

    - Aug시 유의사항 : min_image_overlap, min_bbox_overlap, min_bbox_count, allow_partial_occurrence

    - resize 관련 변환 :: resize_to , keep_asepect_ratio, resize_based_on

    - 그 외 특이한 변수 : max_random_trials

    Glossary(용어설명)
    
    - aspect_ratio :: 화면의 비율을 의미


    """
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100
    ):

        # Rotate_anchors : 회전의 중심이 되는 각도
        # ex) rotate_anchors = None ==> 0으로 간주 
        # ex) rotate_anchors = 100 ==> 100도를 중심으로 움직이겠다는 것
        # ex) rotate_anchors = [-45, 0, 45] => -45, 0, 45도를 중심으로 셋 중 하나를 선택해, 그 후 rotate_range만큼 움직임

        if rotate_anchors is None:
            self.rotate_anchors = None
        elif type(rotate_anchors) in [int, float]:
            self.rotate_anchors = [rotate_anchors]
        else:
            self.rotate_anchors = list(rotate_anchors)


        # Rotate_range : 회전시킬 정도
        # ex) rotate_range = None ==> 0으로 간주 
        # ex) rotate_range = 10 ==> (-10, 10) 이내
        # ex) rotate_range = 30 ==> (30,40) 이내 회전

        if rotate_range is None:
            self.rotate_range = None
        elif type(rotate_range) in [int, float]:
            assert rotate_range >= 0
            self.rotate_range = (-rotate_range, rotate_range)
        elif len(rotate_range) == 2:
            assert rotate_range[0] <= rotate_range[1]
            self.rotate_range = tuple(rotate_range)
        else:
            raise TypeError


        # crop_aspect_ratio : crop할 정도의 비율. 가로 세로의 비율을 의미한다.
        # None => 디폴트값

        if crop_aspect_ratio is None:
            self.crop_aspect_ratio = None
        elif type(crop_aspect_ratio) in [int, float]:
            self.crop_aspect_ratio = float(crop_aspect_ratio)
        elif len(crop_aspect_ratio) == 2:
            self.crop_aspect_ratio = tuple(crop_aspect_ratio)
        else:
            raise TypeError


        # crop_size : crop시킬 사이즈.
        # 300
        # (100,200) -> 가로 100 세로 200으로 자르겠단 것

        if type(crop_size) in [int, float]:
            self.crop_size = crop_size
        elif len(crop_size) == 2:
            assert type(crop_size[0]) == type(crop_size[1])
            self.crop_size = tuple(crop_size)
        else:
            raise TypeError


        # crop_size_by : crop시킬 방법
        # width 중심을 자르냐, height 중심으로 자르냐, 긴 거 중심으로 자르냐 그 차이

        assert crop_size_by in ['width', 'height', 'longest']
        self.crop_size_by = crop_size_by

        self.hflip, self.vflip = hflip, vflip

        self.random_translate = random_translate

        self.min_image_overlap = max(min_image_overlap or 0, 0)
        self.min_bbox_overlap = max(min_bbox_overlap or 0, 0)
        self.min_bbox_count = max(min_bbox_count or 0, 0)
        self.allow_partial_occurrence = allow_partial_occurrence

        self.max_random_trials = max_random_trials

        #resize_to : resize 시킬 사이즈
        #crop하고 resize할건데, 여기서 keep_aspect_ratio(지금의 비율 유지 여부)를 지정해줘야합니다.
        # None => input과 그대로 반환.
        # 100 => keep_aspect_ratio가 False라면 (100,100), True라면 100
        #(100,200) -> 가로 100 세로 200으로 리사이즈하겠단 것. keep_aspect_ratio가 무조건 False여야함 

        if resize_to is None:
            self.resize_to = resize_to
        elif type(resize_to) in [int, float]:
            if not keep_aspect_ratio:
                self.resize_to = (resize_to, resize_to)
            else:
                self.resize_to = resize_to
        elif len(resize_to) == 2:
            assert not keep_aspect_ratio
            assert type(resize_to[0]) == type(resize_to[1])
            self.resize_to = tuple(resize_to)
        assert resize_based_on in ['width', 'height', 'longest']
        self.keep_aspect_ratio, self.resize_based_on = keep_aspect_ratio, resize_based_on


        # crop_rotate_resize가 핵심 로직!
    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        return self.crop_rotate_resize(image, word_bboxes=word_bboxes,
                                       by_word_char_bboxes=by_word_char_bboxes, masks=masks)


    # Theta: 회전시킬 각도 얻기! 
    # rotate_anchors가 있다면(없으면 0으로 지정) 거기서 하나를 고른 후, rotate_range 범위 내의 각도를 더해서 theta 반환
 
    def _get_theta(self):
        if self.rotate_anchors is None:
            theta = 0
        else:
            theta = npr.choice(self.rotate_anchors)
        if self.rotate_range is not None:
            theta += npr.uniform(*self.rotate_range)

        return theta


    #Patch size얻기 :: 이미지의 일부를 patch라고 함
    # crop_aspect_ratio는 crop시킬 이미지의 가로세로비율이고
    # crop_size는 crop시키고자 하는 이미지 패치의 크기



    def _get_patch_size(self, ih, iw):
        if (self.crop_aspect_ratio is None and isinstance(self.crop_size, float) and
            self.crop_size == 1.0):
            return ih, iw


        #별도로 crop_aspect_ratio를 지정안했다면 width / height(가로/세로). 이를 여러개로 지정했다면, crop.aspect_ratio 사이에서 하나를 랜덤하게 지정 

        if self.crop_aspect_ratio is None:
            aspect_ratio = iw / ih
        elif isinstance(self.crop_aspect_ratio, float):
            aspect_ratio = self.crop_aspect_ratio
        else:
            aspect_ratio = np.exp(npr.uniform(*np.log(self.crop_aspect_ratio)))

        #유사하게 crop_size를 지정

        if isinstance(self.crop_size, tuple):
            if isinstance(self.crop_size[0], int):
                crop_size = npr.randint(self.crop_size[0], self.crop_size[1])
            elif self.crop_size[0]:
                crop_size = np.exp(npr.uniform(*np.log(self.crop_size)))
        else:
            crop_size = self.crop_size

        # 우리가 crop을 통해 만들고자 하는 것은 결국 image patch.
        # patch의 h,w는 crop.size_by와 관련이 있다.
        # pw > ph라면, pw : ph => crop_size : ph * (pw/crop_size == aspect_ratio) 로 바뀜(비례식)
    
        if self.crop_size_by == 'longest' and iw >= ih or self.crop_size_by == 'width':
            if isinstance(crop_size, int):
                pw = crop_size
                ph = int(pw / aspect_ratio)
            else:
                pw = int(iw * crop_size)
                ph = int(iw * crop_size / aspect_ratio)
        else:
            if isinstance(crop_size, int):
                ph = crop_size
                pw = int(pw * aspect_ratio)
            else:
                ph = int(ih * crop_size)
                pw = int(ih * crop_size * aspect_ratio)

        return ph, pw

    # patch의 score_map 영역을 지정 :: patch is (2,2) array
    # theta를 통해 얻어낸 회전행렬에 1/4 scale된 quad를 곱해서 얻는다.
    # (-half_width ~ half_width), (-half_height, half_height) 까지 받을 수 있게 지정

    def _get_patch_quad(self, theta, ph, pw):
        cos, sin = np.cos(theta * np.pi / 180), np.sin(theta * np.pi / 180)
        hpx, hpy = 0.5 * pw, 0.5 * ph  # half patch size
        quad = np.array([[-hpx, -hpy], [hpx, -hpy], [hpx, hpy], [-hpx, hpy]], dtype=np.float32)
        rotation_m = np.array([[cos, sin], [-sin, cos]], dtype=np.float32)
        quad = np.matmul(quad, rotation_m)  # patch quadrilateral in relative coords

        return quad

    # input으로부터 괜찮은 patch를 얻어내는 과정
    # 원본 image 직사각형 전체인 image_poly를 선언
    # input으로 받은 patch_quad_rel을 이미지의 center_point만큼 평행이동 ==> 결론적으론 patch_quad_rel이 image patch의 특정 영역을 차지하게 됌
    # ==> 이 patch_quad_rel과 원본 image가 최대한 겹칠 수 있는 영역을 먼저 계산하여, max_available_overlap이라고 정의(이론상 최대 수치)


    def _get_located_patch_quad(self, ih, iw, patch_quad_rel, bboxes=[]):
        image_poly = Polygon([[0, 0], [iw, 0], [iw, ih], [0, ih]])
        if self.min_image_overlap is not None:
            center_patch_poly = Polygon(
                np.array([0.5 * ih, 0.5 * iw], dtype=np.float32) + patch_quad_rel)
            max_available_overlap = (
                image_poly.intersection(center_patch_poly).area / center_patch_poly.area)
            min_image_overlap = min(self.min_image_overlap, max_available_overlap)
        else:
            min_image_overlap = None

        if self.min_bbox_count > 0:
            min_bbox_count = min(self.min_bbox_count, len(bboxes))
        else:
            min_bbox_count = 0



        #(TODO) 회전된 직사각형의 박스들을 기준으로, 3번째 x좌표와 3번째 y좌표를 구한다.(뭔 소리지)
        cx_margin, cy_margin = np.sort(patch_quad_rel[:, 0])[2], np.sort(patch_quad_rel[:, 1])[2]

        found_randomly = False

        # Random하게 cx_margin, cy_margin에서 랜덤하게 가져와서, 그걸 기준으로 patch_poly를 만듬
        for trial_idx in range(self.max_random_trials):
            cx, cy = npr.uniform(cx_margin, iw - cx_margin), npr.uniform(cy_margin, ih - cy_margin)
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            patch_poly = Polygon(patch_quad)
            

            #min_image_overlap일 경우, 전체 image_poly와 현재 patch_poly의 겹치는 영역을 구해서 
            # 그 비율이 min_image_overlap보다 낮은지 확인
            if min_image_overlap:
                image_overlap = patch_poly.intersection(image_poly).area / patch_poly.area
                # 이미지에서 벗어난 영역이 특정 비율보다 높으면 탈락
                if image_overlap < min_image_overlap:
                    continue

            
            # min_box_overlap이 있고, min_bbox_count가 있거나 partial_occurrence를 허용하지 않는다면
            if (self.min_bbox_count or not self.allow_partial_occurrence) and self.min_bbox_overlap:
                bbox_count = 0
                partial_occurrence = False
                
                #bbox에 대해 loop를 돌며, bbox와 patch_poly가 겹치는 정도인 bbox_overlap을 구한다. 
                # bbox_overlap이 min_box_overlap보다 크다면, bbox_count를 늘려간다.
                # 다만, partial occurrence(개체가 부분만 나타나는 것)를 금지했는데, min_overlap보다 작으면 바로 처단.
                for bbox in bboxes:
                    bbox_poly = Polygon(bbox)
                    if bbox_poly.area or patch_poly.area <= 0:
                        continue
                    
                    bbox_overlap = bbox_poly.intersection(patch_poly).area / bbox_poly.area
                    if bbox_overlap >= self.min_bbox_overlap:
                        bbox_count += 1
                    if (not self.allow_partial_occurrence and bbox_overlap > 0 and
                        bbox_overlap < self.min_bbox_overlap):
                        partial_occurrence = True
                        break
                
                # 부분적으로 나타나는 개체가 있으면 탈락
                if partial_occurrence:
                    continue
                # 온전히 포함하는 개체가 특정 개수 미만이면 탈락
                elif self.min_bbox_count and bbox_count < self.min_bbox_count:
                    continue
            
            #조건을 잘 지켰다면, random하게 찾는 방식을 True로 간주한다.
            found_randomly = True
            break

        if found_randomly:
            return patch_quad, trial_idx + 1
        else:
            return None, trial_idx + 1



    #사실상 이 augmentation의 핵심로직.

    def crop_rotate_resize(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[]):
        """
        Args:
            image (ndarray): (H, W, C) shaped ndarray.
            masks (List[ndarray]): List of (H, W) shaped ndarray the same size as the image.
        """
        ih, iw = image.shape[:2]  # image height and width

        # patch를 만들기 위한 theta, patch_size를 받아온다
        theta = self._get_theta()
        ph, pw = self._get_patch_size(ih, iw)

        #받아온 theta, patch_size로 patch_quad_rel을 만든다.
        patch_quad_rel = self._get_patch_quad(theta, ph, pw)


        #translate가 없다면, cx, cy를 중심으로 설정한다.
        if not self.random_translate:
            cx, cy = 0.5 * iw, 0.5 * ih
            patch_quad = np.array([cx, cy], dtype=np.float32) + patch_quad_rel
            num_trials = 0
        
        #translate가 있다면, 여러번 시도하여 조금은 기울어진 patch_quad를 얻어낸다.
        else:
            patch_quad, num_trials = self._get_located_patch_quad(ih, iw, patch_quad_rel,
                                                                  bboxes=word_bboxes)

        # 50% 확률로 vflip, hflip을 먹이겠단 것
        vflip, hflip = self.vflip and npr.randint(2) > 0, self.hflip and npr.randint(2) > 0


        # resize시, None이면 input그대로 반환
        # 지정했는데 keep_aspect_ratio라면, resize_based_on에 비례하게 resize를 해준다.
        if self.resize_to is None:
            oh, ow = ih, iw
        elif self.keep_aspect_ratio:  # `resize_to`: Union[int, float]
            if self.resize_based_on == 'height' or self.resize_based_on == 'longest' and ih >= iw:
                oh = ih * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                ow = int(oh * iw / ih)
            else:
                ow = iw * self.resize_to if isinstance(self.resize_to, float) else self.resize_to
                oh = int(ow * ih / iw)
        elif isinstance(self.resize_to[0], float):  # `resize_to`: tuple[float, float]
            oh, ow = ih * self.resize_to[0], iw * self.resize_to[1]
        else:  # `resize_to`: tuple[int, int]
            oh, ow = self.resize_to


        #아무런 변환없이 rotate도, resize도, output도 그대로라면, 그냥 그대로 적용
        if theta == 0 and (ph, pw) == (ih, iw) and (oh, ow) == (ih, iw) and not (hflip or vflip):
            M = None
            transformed = dict(image=image, word_bboxes=word_bboxes,
                               by_word_char_bboxes=by_word_char_bboxes, masks=masks)
        

        # 그렇지 않다면, 이제 cv2.getPerspectiveTransform을 통해 이동변환행렬을 구한다. 
        #  시작점 4개의 좌표와, 도착점 4개의 좌표를 알고 있다면 그 좌표값을 구할 수 있다.
        #  https://deep-learning-study.tistory.com/200 
        # output의 aspect_ratio와 input의 aspect_ratio 을 비교하여, 필요한만큼 padding을 추가한다.

        else:
            dst = np.array([[0, 0], [ow, 0], [ow, oh], [0, oh]], dtype=np.float32)
            if patch_quad is not None:
                src = patch_quad
            else:
                if ow / oh >= iw / ih:
                    pad = int(ow * ih / oh) - iw
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[-off, 0], [iw + pad - off, 0], [iw + pad - off, ih], [-off, ih]],
                        dtype=np.float32)
                else:
                    pad = int(oh * iw / ow) - ih
                    off = npr.randint(pad + 1)  # offset
                    src = np.array(
                        [[0, -off], [iw, -off], [iw, ih + pad - off], [0, ih + pad - off]],
                        dtype=np.float32)

            if hflip:
                src = src[[1, 0, 3, 2]]
            if vflip:
                src = src[[3, 2, 1, 0]]

            M = cv2.getPerspectiveTransform(src, dst)
            transformed = transform_by_matrix(M, image=image, oh=oh, ow=ow, word_bboxes=word_bboxes,
                                              by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        found_randomly = self.random_translate and patch_quad is not None

        return dict(found_randomly=found_randomly, num_trials=num_trials, matrix=M, **transformed)


class ComposedTransformation:
    def __init__(
        self,
        rotate_anchors=None, rotate_range=None,
        crop_aspect_ratio=None, crop_size=1.0, crop_size_by='longest', hflip=False, vflip=False,
        random_translate=False, min_image_overlap=0, min_bbox_overlap=0, min_bbox_count=0,
        allow_partial_occurrence=True,
        resize_to=None, keep_aspect_ratio=False, resize_based_on='longest', max_random_trials=100,
        brightness=0, contrast=0, saturation=0, hue=0,
        normalize=False, mean=None, std=None, to_tensor=False, to_gray=False
    ):
        self.geo_transform_fn = GeoTransformation(
            rotate_anchors=rotate_anchors, rotate_range=rotate_range,
            crop_aspect_ratio=crop_aspect_ratio, crop_size=crop_size, crop_size_by=crop_size_by,
            hflip=hflip, vflip=vflip, random_translate=random_translate,
            min_image_overlap=min_image_overlap, min_bbox_overlap=min_bbox_overlap,
            min_bbox_count=min_bbox_count, allow_partial_occurrence=allow_partial_occurrence,
            resize_to=resize_to, keep_aspect_ratio=keep_aspect_ratio,
            resize_based_on=resize_based_on, max_random_trials=max_random_trials)

        alb_fns = []
        if brightness > 0 or contrast > 0 or saturation > 0 or hue > 0:
            alb_fns.append(A.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue, p=0.5)) # p 수정

        if to_gray:
            alb_fns.append(A.ToGray(p=0.01))

        if normalize:
            kwargs = dict()
            if mean is not None:
                kwargs['mean'] = mean
            if std is not None:
                kwargs['std'] = std
            alb_fns.append(A.Normalize(**kwargs))

        if to_tensor:
            alb_fns.append(ToTensorV2())

        self.alb_transform_fn = A.Compose(alb_fns)

    def __call__(self, image, word_bboxes=[], by_word_char_bboxes=[], masks=[], height_pad_to=None,
                 width_pad_to=None):
        # TODO Seems that normalization should be performed before padding.

        geo_result = self.geo_transform_fn(image, word_bboxes=word_bboxes,
                                           by_word_char_bboxes=by_word_char_bboxes, masks=masks)

        if height_pad_to is not None or width_pad_to is not None:
            min_height = height_pad_to or geo_result['image'].shape[0]
            min_width = width_pad_to or geo_result['image'].shape[1]
            alb_transform_fn = A.Compose([
                A.PadIfNeeded(min_height=min_height, min_width=min_width,
                              border_mode=cv2.BORDER_CONSTANT,
                              position=A.PadIfNeeded.PositionType.TOP_LEFT),
                self.alb_transform_fn])
        else:
            alb_transform_fn = self.alb_transform_fn
        final_result = alb_transform_fn(image=geo_result['image'])
        del geo_result['image']

        return dict(image=final_result['image'], **geo_result)