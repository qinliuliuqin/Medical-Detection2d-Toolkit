import matplotlib.pyplot as plt
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from detection2d.dataset.object_detection_dataset import ObjectDetectionDataset
from detection2d.utils.data_augmt import *
from detection2d.vis.draw_annotation import draw_annotation, draw_rectangle
from detection2d.utils.image_tools import *

resize_size = [400, 600]

# read training and validation label files
train_label_file = '/mnt/projects/CXR_Object/dataset/train_label.csv'
train_image_folder = '/mnt/projects/CXR_Object/data/train'

labels_tr = pd.read_csv(train_label_file, na_filter=False)
data_folder_tr = train_image_folder
print(f'{len(os.listdir(data_folder_tr))} pictures in {data_folder_tr}')

labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))

# load training dataset
dataset_train = ObjectDetectionDataset(
    data_folder=data_folder_tr,
    data_type='train',
    labels_dict=img_class_dict_tr,
    resize_size=resize_size
)

idx = 12
img, target = dataset_train.__getitem__(idx)
boxes = target['boxes']

img = transforms.ToPILImage()(img)

# numpy shape [C, H, W]
img_npy = img.numpy()
img = convert_to_PIL_image(img_npy)
boxes = [list(box) for box in boxes.numpy().astype(dtype=np.int32)]

img = transforms.ToPILImage()(img_tensor)
draw_rectangle(img, boxes)
plt.imshow(img)
plt.show()
#
#
# # convert numpy to [H,W,C]
# transposed_img_npy = np.transpose(img_npy, (1, 2, 0))
# transposed_boxes = []
# for box in boxes:
#     extended_box = box[:]
#     extended_box.append(1)
#     transposed_boxes.append(extended_box)
# transposed_boxes = np.array(transposed_boxes, dtype=np.int32)
#
# transposed_img_npy, transposed_boxes = RandomHorizontalFlip()(transposed_img_npy, transposed_boxes)
#
# # convert numpy to the original shape [C, H, W]
# flipped_img_npy = np.transpose(transposed_img_npy, (2, 0, 1))
# boxes = []
# for box in transposed_boxes:
#     boxes.append(box[:4])
#
# img = Image.fromarray(img_npy)
# draw_rectangle(img, boxes)
#
#
# plt.imshow(img)
# plt.show()


# # test horizontal flip
# img_npy, boxes_npy = img.numpy(), boxes.numpy()
# # flipped_img_npy, flipped_boxes_npy = HorizontalFlip()(img_npy, boxes_npy)
#
# from detection2d.utils.bbox_utils import draw_rect
# import matplotlib.pyplot as plt
#
# img_npy = np.transpose(img_npy, (1, 2, 0))
#
#
# image = draw_rect(img_npy, boxes_npy)
# plt.imshow(image)
# plt.show()
