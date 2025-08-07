import os
import json
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose
from torch.utils.data import Dataset
from PIL import Image

from mmengine.registry import build_from_cfg
from cmae.registry import DATASETS, TRANSFORMS


@DATASETS.register_module()
class CMAEDatasetFolder(Dataset):
    def __init__(self, data_root, data_ann=None, pipeline=None, pixel=31, test=False):
        self.data_root = os.path.join(data_root)
        self.test = test

        # Use ImageFolder instead of JSON annotation
        self.image_folder = ImageFolder(root=data_root)
        self.data_infors = self._convert_imagefolder_to_annotations(self.image_folder)

        # Compose base pipeline (e.g. load image, resize, normalize)
        pipeline_base = [build_from_cfg(p, TRANSFORMS) for p in pipeline[:3]]
        self.pipeline_base = Compose(pipeline_base)

        # Compose final pipeline
        pipeline_final = [build_from_cfg(p, TRANSFORMS) for p in pipeline[3:]]
        self.shift = build_from_cfg(dict(type='ShiftPixel', pixel=0), TRANSFORMS)
        self.pipeline_final = Compose(pipeline_final)

        # Compose augmentation pipeline
        pipeline_aug = [
            dict(type='ShiftPixel', pixel=pixel),
            dict(
                type='RandomApply',
                transforms=[dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                prob=0.8),
            dict(type='RandomGrayscale', prob=0.2, keep_channels=True, channel_weights=(0.114, 0.587, 0.2989)),
            dict(type='GaussianBlur', magnitude_range=(0.1, 2.0), magnitude_std='inf', prob=0.5)
        ]
        pipeline_aug_l = [build_from_cfg(p, TRANSFORMS) for p in pipeline_aug]
        self.pipeline_aug = Compose(pipeline_aug_l)

    def _convert_imagefolder_to_annotations(self, image_folder):
        """Converts ImageFolder dataset to annotation-like dicts."""
        data_infos = []
        for img_path, label in image_folder.samples:
            split_img_path = img_path.split(os.sep)
            data_root = "".join(split_img_path[:-2])
            data_infos.append({'img_path': img_path, 'label': label, 'data_root':data_root, 'prefix':split_img_path[-2], 'filename':split_img_path[-1]})
        return data_infos

    def __len__(self):
        return len(self.data_infors)

    def __getitem__(self, idx):

        item = self.data_infors[idx].copy()
        item['data_root'] = self.data_root
        src_img = self.pipeline_base(item)

        patch_results = {'img': src_img['img']}
        img_t_results = {'img': src_img['img'].copy()}

        patch = self.pipeline_final(self.shift(patch_results))

        img_t = self.pipeline_final(self.pipeline_aug(img_t_results))

        out = {'img':patch['img'],'img_t':img_t['img']}

        return out