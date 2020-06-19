from easydict import EasyDict as edict

from detection2d.utils.data_augmt import *


__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.train_label_file = '/mnt/projects/CXR_Object/dataset/train_label_debug.csv'

__C.general.train_image_folder = '/mnt/projects/CXR_Object/data/train'

__C.general.val_label_file = '/mnt/projects/CXR_Object/dataset/dev_label_debug.csv'

__C.general.val_image_folder = '/mnt/projects/CXR_Object/data/dev'

__C.general.save_dir = '/mnt/projects/CXR_Object/models/model_0618_2020_debug'

__C.general.resume_epoch = -1

__C.general.num_gpus = 0

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.resize_size = [600, 600]

##################################
# data augmentation parameters
##################################

__C.data_augmentations = [
    RandomHorizontalFlip(),
    RandomTranslate()
]

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

__C.train.epochs = 101

__C.train.batch_size = 1

__C.train.num_threads = 1

__C.train.lr = 0.005

__C.train.save_epochs = 100

__C.train.print_freq = 1

##################################
# optimizer parameters
##################################

__C.train.optimizer = {}

__C.train.optimizer.name = 'SGD' # 'SGD' or 'Adam'

__C.train.optimizer.momentum = 0.9  # used for SGD

__C.train.optimizer.betas = (0.9, 0.999)  # used for Adam

__C.train.optimizer.weight_decay = 0.0005

##################################
# scheduler parameters
##################################

__C.train.scheduler = {}

__C.train.scheduler.step_size = 5

__C.train.scheduler.gamma = 0.1

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
