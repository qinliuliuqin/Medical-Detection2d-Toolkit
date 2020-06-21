from easydict import EasyDict as edict


__C = edict()
cfg = __C

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.resize_size = [600, 600]

__C.dataset.normalizer = {'Adaptive': None}  # {'Fixed': {'mean': [0, 0, 0], 'std': [1, 1, 1]}}

