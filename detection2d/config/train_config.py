from easydict import EasyDict as edict
import torchvision.transforms as transforms


__C = edict()
cfg = __C

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

# sampling method:
# 1) GLOBAL: sampling crops randomly in the entire image domain
__C.dataset.sampling_method = 'GLOBAL'

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

##################################
# data augmentation parameters
##################################
__C.data_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

##################################
# loss function
##################################
__C.landmark_loss = {}

__C.landmark_loss.name = 'Focal'          # 'Dice', or 'Focal'

__C.landmark_loss.focal_obj_alpha = [0.75] * 24  # class balancing weight for focal loss

__C.landmark_loss.focal_gamma = 2         # gamma in pow(1-p,gamma) for focal loss

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
