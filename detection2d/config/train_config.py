from easydict import EasyDict as edict
import torchvision.transforms as transforms


__C = edict()
cfg = __C

WIDTH = 600
HEIGHT = 600

##################################
# general parameters
##################################
__C.general = {}

__C.general.train_label_file = '/shenlab/lab_stor6/projects/CXR_Object/train.csv'

__C.general.train_image_folder = '/shenlab/lab_stor6/projects/CXR_Object/train'

__C.general.val_label_file = '/shenlab/lab_stor6/projects/CXR_Object/dev.csv'

__C.general.val_image_folder = '/shenlab/lab_stor6/projects/CXR_Object/dev'

__C.general.save_dir = '/mnt/projects/CXR_Object/models/model_0610_2020'

__C.general.resume_epoch = -1

__C.general.num_gpus = 1

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.resize_size = [WIDTH, HEIGHT]

##################################
# data augmentation parameters
##################################
__C.data_transforms = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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

__C.train.epochs = 101

__C.train.batch_size = 12

__C.train.num_threads = 12

__C.train.lr = 0.005

__C.train.save_epochs = 100

__C.train.print_freq = 20

##################################
# optimizer parameters
##################################

__C.train.optimizer = {}

__C.train.optimizer.name = 'SGD' # 'SGD' or 'Adam'

__C.train.optimizer.sgd_momentum = 0.9

__C.train.optimizer.adam_betas = (0.9, 0.999)

__C.train.optimizer.weight_decay = 0.0005

__C.train.optimizer.step_size = 5

__C.train.optimizer.gamma = 0.1

##################################
# debug parameters
##################################
__C.debug = {}

# random seed used in training
__C.debug.seed = 0

# whether to save input crops
__C.debug.save_inputs = False
