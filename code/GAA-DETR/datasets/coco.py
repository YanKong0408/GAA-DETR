# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
import os.path
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, mode = 'val'):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.mode = mode
        self.prepare = ConvertCocoPolysToMask(return_masks)
    
    # load heatmap
    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB"), path

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def _load_attention_map(self, id: int) -> np.ndarray:
        path = self.coco.loadImgs(id)[0]["file_name"].replace('.jpg', '_heatmap.jpg')
        heatmap_path = os.path.join(self.root, path)
        return Image.open(heatmap_path).convert("L")
    
    def __getitem__(self, idx):
        # if self.mode != 'train':
        #     img, target = super(CocoDetection, self).__getitem__(idx)
        #     image_id = self.ids[idx]
        #     target = {'image_id': image_id, 'annotations': target}
        #     img, target = self.prepare(img, target)
        #     if self._transforms is not None:
        #         img, target, _ = self._transforms(img, target, None)
        #     # img_v = img.numpy()
        #     # cv2.imshow('',img_v) 
        #     # cv2.waitKey(0)
        #     return img, target
        #  # ========================================= difference ================================================= #
        # else:
        if not isinstance(idx, int):
            raise ValueError(f"Index must be of type integer, got {type(idx)} instead.")

        # load
        id = self.ids[idx]
        img, img_path = self._load_image(id)
        attention_map = self._load_attention_map(id)

        target = self._load_target(id)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target, heatmap = self._transforms(img, target, attention_map)
        target['image_file'] = img_path
        # print('img',img.size())
        # print('heatmap',heatmap.size())

        # img_v = img.permute(1, 2, 0).cpu().numpy()
        # cv2.imshow('img',img_v)
        # cv2.waitKey(0)

        # heatmap_v = heatmap.permute(1, 2, 0).cpu().numpy()
        # cv2.imshow('heatmap',heatmap_v)
        # cv2.waitKey(0)

        # decompose and match heatmap
        heatmap = heatmap.numpy()
        bboxes = target['boxes'].numpy()
        _, W, H =heatmap.shape
        heatmap = heatmap.reshape(W, H)

        # decompose
        binary_map = heatmap > 0
        num_labels, labels = cv2.connectedComponents(binary_map.astype(np.uint8))

        regions_array = np.zeros((num_labels - 1, W, H), dtype=np.float32)
        for i in range(1, num_labels):  
            regions_array[i - 1] = (labels == i).astype(np.float32) * heatmap 

        # remove small areas
        row_sums = regions_array.sum(axis=(1, 2)) 
        valid_rows = row_sums >= 500
        regions_array = regions_array[valid_rows]

        # match
        n = regions_array.shape[0]
        N = bboxes.shape[0]  
        areas = np.zeros((n, N), dtype=np.float32)
        for j, bbox in enumerate(bboxes):
            xc, yc, w, h = bbox
            x_min = int((xc - 0.5 * w) * H) 
            y_min = int((yc - 0.5 * h) * W)
            x_max = int((xc + 0.5 * w) * H) 
            y_max = int((yc + 0.5 * h) * W)
            bbox_mask = np.zeros((W, H), dtype=np.float32)
            bbox_mask[y_min:y_max, x_min:x_max] = 1.0

            for i in range(n):
                intersection = regions_array[i] * bbox_mask  
                areas[i, j] = intersection.sum()  
        sum_region = np.sum(regions_array, axis=(1, 2))
        area_ra = areas / sum_region[:, np.newaxis]

        # get gaze-object heatmap
        
        gaze_object_heatmap = np.zeros((N, W, H))
        indexes = []
        for j in range(area_ra.shape[1]):
            column = area_ra[:, j]
            
            index = np.where(column > 0.5)[0]
            index2 = np.where(column > 0.05)[0]
            if len(index)!=0:
                for i in index:
                    gaze_object_heatmap[j] += regions_array[i]
                    indexes.append(i)
            elif len(index2) !=0:
                        max_index = np.argmax(column)
                        gaze_object_heatmap[j] += regions_array[max_index]
                        indexes.append(max_index)
        target['target-gaze_heatmap'] = torch.tensor(gaze_object_heatmap)

        selected_index = np.setdiff1d(range(n), indexes)
        if len(selected_index) != 0:
            gaze_only_heatmap = regions_array[selected_index]
            target['gaze-only_heatmap'] = torch.tensor(gaze_only_heatmap)
        else:
            target['gaze-only_heatmap'] = torch.zeros(1, W, H)         
        
        return img, target
    
        # ========================================= difference end ============================================= #
        


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize_inf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    normalize_train = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize_train,
        ])
    if image_set == 'val' or image_set=='test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize_inf,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'{mode}_test2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks, mode=image_set)
    return dataset
