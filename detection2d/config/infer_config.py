from easydict import EasyDict as edict

from detection2d.utils.data_augmt import *


__C = edict()
cfg = __C

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.resize_size = [600, 600]

__C.dataset.normalizer = {'Fixed': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}

##################################
# data augmentation parameters
##################################
__C.augmentations = augmentation([
    # RandomNegativeAndPositiveFlip(),
    # RandomHorizontalFlip(),
    # RandomVerticalFlip(),
    # RandomTranspose()
])
