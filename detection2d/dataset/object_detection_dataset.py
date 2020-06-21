import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt
from detection2d.utils.image_tools import normalize


class ObjectDetectionDataset(object):

    def __init__(self, data_folder, data_type, labels_dict, resize_size, normalizer, augmentations=None):
        """

        :param data_folder:
        :param data_type: only support three data types, namely 'train', 'val', and 'test'.
        :param labels_dict:
        :param resize_size:
        """
        self.data_folder = data_folder
        self.data_type = data_type
        self.labels_dict = labels_dict
        self.resize_size = resize_size
        self.normalizer = normalizer
        self.image_files_list = [s for s in sorted(os.listdir(data_folder)) if s in labels_dict.keys()]
        self.augmentations = augmentations
        self.annotations = [labels_dict[i] for i in self.image_files_list]

    def __getitem__(self, idx):
        # load images
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.data_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]

        if self.data_type == 'train':
            annot_boxes_coords, annot_boxes_labels = \
                read_boxes_from_annotation_txt(self.labels_dict[img_name], [width, height], self.resize_size)

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size[::-1])(img)

            img = transforms.ToTensor()(img)

            if self.normalizer is not None:
                img = normalize(img, self.normalizer)

            # convert the coordinates and labels of the annotated boxes to torch.Tensor
            annot_boxes_coords = torch.as_tensor(annot_boxes_coords, dtype=torch.float32)
            annot_boxes_labels = torch.ones((len(annot_boxes_coords),), dtype=torch.int64)

            image_id = torch.tensor([idx])
            annot_boxes_area = (annot_boxes_coords[:, 3] - annot_boxes_coords[:, 1]) * \
                               (annot_boxes_coords[:, 2] - annot_boxes_coords[:, 0])

            # suppose all instances are not crowd
            is_crowd = torch.zeros((len(annot_boxes_coords),), dtype=torch.int64)

            target = {
                "boxes": annot_boxes_coords,
                "labels": annot_boxes_labels,
                "image_id": image_id,
                "area": annot_boxes_area,
                "is_crowd": is_crowd
            }

            return img, target

        elif self.data_type == 'val':
            label = 0 if self.labels_dict[img_name] == '' else 1

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            img = transforms.ToTensor()(img)

            if self.normalizer is not None:
                img = normalize(img, self.normalizer)

            return img, label, width, height

        elif self.data_type == 'test':
            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            return transforms.ToTensor()(img), width, height

        else:
            raise ValueError('Unsupported dataset type!')

    def __len__(self):
        return len(self.image_files_list)
