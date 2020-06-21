import argparse
import pandas as pd
import os
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc

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


def infer(model_folder, data_folder, infer_file, num_classes, save_folder, cuda_id):
    """
    The inference interface
    :param model_folder:
    :param model_name:
    :param cuda_id:
    :param num_classes:
    :param save_folder:
    :return:
    """
    # load model
    device = get_device(cuda_id)
    model = get_detection_model(num_classes, False)
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'model.pt'), map_location=device))
    model.eval()

    infer_cfg = load_config(os.path.join(model_folder, 'infer_config.py'))
    resize_size = infer_cfg.dataset.resize_size
    normalizer = infer_cfg.dataset.normalizer

    labels_df = pd.read_csv(infer_file, na_filter=False)
    labels_dict = dict(zip(labels_df.image_name, labels_df.annotation))

    preds, labels, locs = [], [], []

    # load test dataset
    dataset = ObjectDetectionDataset(
        data_folder=data_folder,
        data_type='val',
        labels_dict=labels_dict,
        resize_size=resize_size,
        normalizer=normalizer
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    with torch.no_grad():
        for image, label, width, height in tqdm(data_loader):

            image = list(img.to(device) for img in image)
            labels.append(label[-1])

            outputs = model(image)

            center_points = []
            center_points_preds = []

            if len(outputs[-1]['boxes']) == 0:
                preds.append(0)
                center_points.append([])
                center_points_preds.append('')
                locs.append('')
            else:
                preds.append(torch.max(outputs[-1]['scores']).tolist())

                new_output_index = torch.where((outputs[-1]['scores'] > 0.1))
                new_boxes = outputs[-1]['boxes'][new_output_index]
                new_scores = outputs[-1]['scores'][new_output_index]

                for i in range(len(new_boxes)):
                    new_box = new_boxes[i].tolist()
                    center_x = (new_box[0] + new_box[2]) / 2
                    center_y = (new_box[1] + new_box[3]) / 2
                    if resize_size is not None:
                        center_points.append([center_x / resize_size[0] * width[-1], center_y / resize_size[1] * height[-1]])
                    else:
                        center_points.append([center_x, center_y])
                center_points_preds += new_scores.tolist()

                line = ''
                for i in range(len(new_boxes)):
                    if i == len(new_boxes) - 1:
                        line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(
                            center_points[i][1])
                    else:
                        line += str(center_points_preds[i]) + ' ' + str(center_points[i][0]) + ' ' + str(
                            center_points[i][1]) + ';'
                locs.append(line)

    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    cls_res = pd.DataFrame({'image_name': dataset.image_files_list, 'prediction': preds})
    cls_res.to_csv(
        os.path.join(save_folder, 'classification.csv'), columns=['image_name', 'prediction'], sep=',', index=None
    )
    print('classification.csv generated.')

    loc_res = pd.DataFrame({'image_name': dataset.image_files_list, 'prediction': locs})
    loc_res.to_csv(
        os.path.join(save_folder, 'localization.csv'), columns=['image_name', 'prediction'], sep=',', index=None
    )
    print('localization.csv generated.')

    pred = cls_res.prediction.values
    gt = labels_df.annotation.astype(bool).astype(float).values

    acc = ((pred >= .5) == gt).mean()
    fpr, tpr, _ = roc_curve(gt, pred)
    roc_auc = auc(fpr, tpr)
    print('ACC: {}'.format(acc), 'AUC: {}'.format(roc_auc))


def main():
    long_description = "Inference engine for 2d medical image object detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-m', '--model-folder',
                        default='/shenlab/lab_stor6/qinliu/projects/CXR_Object/models/model_0620_2020/contrast_fixed_normalizer',
                        help='Model folder containing the model and inference config file.')
    parser.add_argument('-d', '--data-folder',
                        default='/shenlab/lab_stor6/projects/CXR_Object/dev',
                        help='The data folder.')
    parser.add_argument('-i', '--infer-file',
                        default='/shenlab/lab_stor6/projects/CXR_Object/dev.csv',
                        help='')
    parser.add_argument('-n', '--num-classes',
                        default=2,
                        help='')
    parser.add_argument('-o', '--output',
                        default='/shenlab/lab_stor6/qinliu/projects/CXR_Object/results/model_0620_2020/contrast_fixed_normalizer/dev',
                        help='')
    parser.add_argument('-g', '--gpu',
                        default=6,
                        help='')
    args = parser.parse_args()

    infer(args.model_folder, args.data_folder, args.infer_file, args.num_classes, args.output, args.gpu)


if __name__ == '__main__':

    main()
