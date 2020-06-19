import os
from PIL import Image
import torch
import torchvision.transforms as transforms


class ObjectDetectionDataset(object):

    def __init__(self, data_folder, data_type, labels_dict, resize_size, augmentations=None):
        """

        :param data_folder:
        :param data_type: only support three data types, namely 'train', 'val', and 'test'.
        :param labels_dict:
        :param resize_size:
        :param transforms:
        """
        self.data_folder = data_folder
        self.data_type = data_type
        self.labels_dict = labels_dict
        self.resize_size = resize_size
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
            annot_boxes_coords, annot_boxes_labels = [], []

            multiple_annots_txt = self.labels_dict[img_name]
            if type(multiple_annots_txt) == str:
                multiple_annots_txt = multiple_annots_txt.split(';')
                for single_annot_txt in multiple_annots_txt:
                    x, y = [], []
                    single_annot_coords = single_annot_txt[2:].split(' ')
                    for i in range(len(single_annot_coords)):
                        if i % 2 == 0:
                            x.append(float(single_annot_coords[i]))
                        else:
                            y.append(float(single_annot_coords[i]))

                    xmin, xmax = min(x) / width, max(x) / width
                    ymin, ymax = min(y) / height, max(y) / height

                    if self.resize_size is not None:
                        xmin, xmax = xmin * self.resize_size[0], xmax * self.resize_size[0]
                        ymin, ymax = ymin * self.resize_size[1], ymax * self.resize_size[1]

                    annot_boxes_coords.append([xmin, ymin, xmax, ymax])

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

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            if self.augmentations is not None:
                for augmentation in self.augmentations:
                    img, annot_boxes_coords = augmentation(img, annot_boxes_coords)

            return img, target

        elif self.data_type == 'val':
            label = 0 if self.labels_dict[img_name] == '' else 1

            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            return img, label, width, height

        elif self.data_type == 'test':
            if self.resize_size is not None:
                img = transforms.Resize(self.resize_size)(img)

            return img, width, height

        else:
            raise ValueError('Unsupported dataset type!')

    def __len__(self):
        return len(self.image_files_list)