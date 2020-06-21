import os
from PIL import Image
import torch
import torchvision.transforms as transforms

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt


class ObjectDetectionDataset(object):

    def normalize(self, image, normalizer):
        """
        Normalize the input image given the normalizer
        :param image:
        :param normalizer:
        :return:
        """
        if 'Fixed' in normalizer:
            mean, std = normalizer['Fixed']['mean'], normalizer['Fixed']['std']
            image = transforms.Normalize(mean=mean, std=std)(image)

        if 'Adaptive' in normalizer:
            mean, std = torch.mean(image, dim=[1, 2]), torch.std(image, dim=[1, 2])
            image = transforms.Normalize(mean=list(mean.numpy()), std=list(std.numpy()))(image)

        return image

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
            boxes, labels = read_boxes_from_annotation_txt(
                self.labels_dict[img_name], [width, height], self.resize_size)

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size[::-1])(img)

            if self.augmentations is not None:
                img, boxes = self.augmentations(img, boxes)

            img = transforms.ToTensor()(img)

            if self.normalizer is not None:
                img = self.normalize(img, self.normalizer)

            # convert the coordinates and labels of the annotated boxes to torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            image_id = torch.tensor([idx])
            annot_boxes_area = (boxes[:, 3] - boxes[:, 1]) * \
                               (boxes[:, 2] - boxes[:, 0])

            # suppose all instances are not crowd
            is_crowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {"boxes": boxes, "labels": labels,
                      "image_id": image_id, "area": annot_boxes_area, "is_crowd": is_crowd}

            return img, target

        elif self.data_type == 'val':
            label = 0 if self.labels_dict[img_name] == '' else 1

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            img = transforms.ToTensor()(img)

            if self.normalizer is not None:
                img = self.normalize(img, self.normalizer)

            return img, label, width, height

        elif self.data_type == 'test':
            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            return transforms.ToTensor()(img), width, height

        else:
            raise ValueError('Unsupported dataset type!')

    def __len__(self):
        return len(self.image_files_list)
