from easydict import EasyDict as edict
from detection3d.utils.normalizer import FixedNormalizer, AdaptiveNormalizer

__C = edict()
cfg = __C

##################################
# general parameters
##################################
__C.general = {}

__C.general.train_label_file = '/mnt/projects/CXR_Object/train.csv'

__C.general.train_image_folder = '/mnt/projects/CXR_Object/train'

__C.general.val_label_file = '/mnt/projects/CXR_Object/dev.csv'

__C.general.val_image_folder = '/mnt/projects/CXR_Object/dev'

__C.general.save_dir = '/mnt/projects/CXR_Object/models/model_0607_2020'

__C.general.resume_epoch = -1

__C.general.num_gpus = 0

##################################
# dataset parameters
##################################
__C.dataset = {}

__C.dataset.num_classes = 2

__C.dataset.crop_spacing = [2, 2, 2]      # mm

__C.dataset.crop_size = [96, 96, 96]   # voxel

__C.dataset.sampling_size = [6, 6, 6]      # voxel

__C.dataset.positive_upper_bound = 3    # voxel

__C.dataset.negative_lower_bound = 6    # voxel

__C.dataset.num_pos_patches_per_image = 8

__C.dataset.num_neg_patches_per_image = 16

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer()]

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
__C.augmentation = {}

__C.augmentation.turn_on = False

__C.augmentation.orientation_axis = [0, 0, 0]  # [x,y,z], axis = [0,0,0] will set it as random axis.

__C.augmentation.orientation_radian = [-30, 30]  # range of rotation in degree, 1 degree = 0.0175 radian

__C.augmentation.translation = [10, 10, 10]  # mm

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

##################################
# training parameters
##################################
__C.train = {}

__C.train.epochs = 2001

__C.train.batch_size = 1

__C.train.num_threads = 1

__C.train.lr = 0.005

__C.train.save_epochs = 100

__C.train.print_freq = 20

##################################
# optimizer parameters
##################################

__C.train.optimizer = {}

__C.train.optimizer.name = 'Adam' # 'SGD' or 'Adam'

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
