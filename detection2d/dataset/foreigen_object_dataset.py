import os
from PIL import Image
import torch


class ForeignObjectDataset(object):

    def __init__(self, data_folder, data_type, labels_dict, resize_size, transform=None):
        self.data_folder = data_folder
        self.data_type = data_type
        self.labels_dict = labels_dict
        self.resize_size = resize_size
        self.image_files_list = [s for s in sorted(os.listdir(data_folder)) if s in labels_dict.keys()]
        self.transform = transform
        self.annotations = [labels_dict[i] for i in self.image_files_list]

    def __getitem__(self, idx):
        # load images
        img_name = self.image_files_list[idx]
        img_path = os.path.join(self.data_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        width, height = img.size[0], img.size[1]

        if self.data_type == 'train':
            annotation = self.labels_dict[img_name]

            boxes = []
            if type(annotation) == str:
                annotation_list = annotation.split(';')
                for anno in annotation_list:
                    x = []
                    y = []

                    anno = anno[2:]
                    anno = anno.split(' ')
                    for i in range(len(anno)):
                        if i % 2 == 0:
                            x.append(float(anno[i]))
                        else:
                            y.append(float(anno[i]))

                    xmin = min(x) / width * self.resize_size[0]
                    xmax = max(x) / width * self.resize_size[0]
                    ymin = min(y) / height * self.resize_size[1]
                    ymax = max(y) / height * self.resize_size[1]
                    boxes.append([xmin, ymin, xmax, ymax])

            # convert everything into a torch.Tensor
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            # there is only one class
            labels = torch.ones((len(boxes),), dtype=torch.int64)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            if self.transform is not None:
                img = self.transform(img)

            return img, target

        elif self.data_type == 'dev':

            label = 0 if self.labels_dict[img_name] == '' else 1

            if self.transform is not None:
                img = self.transform(img)

            return img, label, width, height

        else:
            raise ValueError('Unsupported dataset type!')

    def __len__(self):
        return len(self.image_files_list)