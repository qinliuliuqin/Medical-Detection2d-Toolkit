from easydict import EasyDict as edict

from detection2d.utils.data_augmt import *


__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.train_label_file = '/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/dataset/train_label.csv'

__C.general.train_image_folder = '/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/stage_2_train_images'

__C.general.val_label_file = '/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/dataset/dev_label.csv'

__C.general.val_image_folder = '/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/stage_2_train_images'

__C.general.save_dir = '/shenlab/lab_stor6/qinliu/projects/CXR_Pneumonia/models/model_0904_2020/normal'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.resize_size = [800, 800]

__C.dataset.normalizer = {'Fixed': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}}

##################################
# data augmentation parameters
##################################
__C.augmentations = augmentation([
    RandomNegativeAndPositiveFlip(),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomTranspose()
])


##################################
# net
##################################
__C.net = {}

__C.net.name = 'faster_rcnn'

__C.net.pre_trained = True

##################################
# training parameters
##################################
__C.train = {}

__C.train.epochs = 301

__C.train.batch_size = 12

__C.train.num_threads = 12

__C.train.lr = 0.0001

__C.train.print_freq = 20

##################################
# optimizer parameters
##################################

__C.train.optimizer = {}

__C.train.optimizer.name = 'Adam' # 'SGD' or 'Adam'

__C.train.optimizer.momentum = 0.9  # used for SGD

__C.train.optimizer.betas = (0.9, 0.999)  # used for Adam

__C.train.optimizer.weight_decay = 0.0

##################################
# scheduler parameters
##################################

__C.train.scheduler = {}

__C.train.scheduler.step_size = 5

__C.train.scheduler.gamma = 0.9

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
