import argparse
import pandas as pd
import os
import torch
from tqdm import tqdm

from detection2d.utils.train_utils import collate_fn
from detection2d.utils.file_io import load_config
from detection2d.dataset.object_detection_dataset import ObjectDetectionDataset
from detection2d.network.faster_rcnn import get_detection_model


def get_device(cuda_id):
    """
    Get device for inference
    :param cuda_id (int): the device id
    :return:
        torch.device type
    """
    if cuda_id >= 0:
        torch.cuda.set_device(cuda_id)
        device = torch.device('cuda:{}'.format(cuda_id))
    else:
        device = torch.device('cpu')

    return device


def infer(model_folder, data_folder, infer_file, num_classes, threshold, save_folder, cuda_id):
    """
    The inference interface
    :param model_folder:
    :param model_name:
    :param cuda_id:
    :param num_classes:
    :param threshold:
    :param save_folder:
    :return:
    """
    # load model
    device = get_device(cuda_id)
    model = get_detection_model(num_classes, False, image_mean=[0, 0, 0], image_std=[1, 1, 1])
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pt'), map_location=device))
    model.eval()

    infer_cfg = load_config(os.path.join(model_folder, 'infer_config.py'))
    resize_size = infer_cfg.dataset.resize_size
    normalizer = infer_cfg.dataset.normalizer
    augmentation = infer_cfg.augmentations

    image_names_df = pd.read_csv(infer_file, na_filter=False)
    image_names_dict = dict(zip(image_names_df.image_name, [0] * len(image_names_df.image_name)))

    preds, labels, centers, locs = [], [], [], []

    # load test dataset
    dataset = ObjectDetectionDataset(
        data_folder=data_folder,
        data_type='test',
        labels_dict=image_names_dict,
        resize_size=resize_size,
        normalizer=normalizer,
        augmentations=augmentation
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for image, width, height in tqdm(data_loader):

            image = list(img.to(device) for img in image)
            outputs = model(image)

            center_points = []
            center_points_preds = []

            if len(outputs[-1]['boxes']) == 0:
                preds.append(0)
                center_points.append([])
                center_points_preds.append('')
                centers.append('')
                locs.append('')
            else:
                max_pred = torch.max(outputs[-1]['scores']).tolist()
                if max_pred < threshold:
                    preds.append(0)
                    center_points.append([])
                    center_points_preds.append('')
                    centers.append('')
                    locs.append('')
                    continue

                preds.append(max_pred)
                new_output_index = torch.where((outputs[-1]['scores'] > float(threshold)))
                new_boxes = outputs[-1]['boxes'][new_output_index]
                new_scores = outputs[-1]['scores'][new_output_index]

                boxes_preds = []
                for i in range(len(new_boxes)):
                    new_box = new_boxes[i].tolist()
                    x_min, y_min = min(new_box[0], new_box[2]), min(new_box[1], new_box[3])
                    x_max, y_max = max(new_box[0], new_box[2]), max(new_box[1], new_box[3])

                    if resize_size is not None:
                        x_min = x_min / resize_size[0] * width[-1]
                        x_max = x_max / resize_size[0] * width[-1]
                        y_min = y_min / resize_size[1] * height[-1]
                        y_max = y_max / resize_size[1] * height[-1]

                    boxes_preds.append([x_min, y_min, x_max, y_max])

                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    center_points.append([center_x, center_y])
                center_points_preds += new_scores.tolist()

                line_center, line_loc = '', ''
                for i in range(len(new_boxes)):
                    x_min, y_min, x_max, y_max = boxes_preds[i]
                    if i == len(new_boxes) - 1:
                        line_center += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1])
                        line_loc += str(center_points_preds[i]) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max)
                    else:
                        line_center += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(center_points[i][1]) + ';'
                        line_loc += str(center_points_preds[i]) + ' ' + str(x_min) + ' ' + str(y_min) + ' ' + str(x_max) + ' ' + str(y_max) + ';'

                centers.append(line_center)
                locs.append(line_loc)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    cls_res = pd.DataFrame({'image_name': dataset.image_files_list, 'prediction': preds})
    cls_res.to_csv(
        os.path.join(save_folder, 'classification.csv'), columns=['image_name', 'prediction'], sep=',', index=None
    )
    print('classification.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset.image_files_list, 'prediction': centers, 'annotation': locs})
    loc_res.to_csv(
        os.path.join(save_folder, 'localization.csv'), columns=['image_name', 'prediction', 'annotation'], sep=',', index=None
    )
    print('localization.csv generated.')


def main():
    long_description = "Inference engine for 2d medical image object detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-m', '--model-folder',
                        default='/shenlab/lab_stor6/qinliu/projects/CXR_Pneumonia/models/model_0904_2020/normal',
                        help='Model folder containing the model and inference config file.')
    parser.add_argument('-d', '--data-folder',
                        default='/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/stage_2_train_images',
                        help='The data folder.')
    parser.add_argument('-i', '--infer-file',
                        default='/shenlab/lab_stor6/qinliu/CXR_Pneumonia/Stage2/dataset/test_label.csv',
                        help='')
    parser.add_argument('-n', '--num-classes',
                        default=2,
                        help='')
    parser.add_argument('-t', '--threshold',
                        default=0.8,
                        help='')
    parser.add_argument('-o', '--output',
                        default='/shenlab/lab_stor6/qinliu/projects/CXR_Pneumonia/results/model_0904_2020/normal/test',
                        help='')
    parser.add_argument('-g', '--gpu',
                        default=0,
                        help='')
    args = parser.parse_args()

    infer(args.model_folder, args.data_folder, args.infer_file, args.num_classes, args.threshold, args.output, args.gpu)


if __name__ == '__main__':

    main()
