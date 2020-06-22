import argparse
import importlib
import numpy as np
import os
import pandas as pd
import shutil
from sklearn.metrics import roc_auc_score
import torch

from detection2d.dataset.object_detection_dataset import ObjectDetectionDataset
from detection2d.core.engine import train_one_epoch
from detection2d.utils import train_utils, file_io


def train(config_file, gpu_id):
    """ Medical image segmentation training engine
    :param config_file: the absolute path of the input configuration file
    :param gpu_id: the gpu id
    :return: None
    """
    assert os.path.isfile(config_file), 'Config not found: {}'.format(config_file)

    # load config file
    cfg = file_io.load_config(config_file)

    # clean the existing folder if training from scratch
    if os.path.isdir(cfg.general.save_dir) and cfg.general.resume_epoch < 0:
        shutil.rmtree(cfg.general.save_dir)

    # create save folder if it does not exist
    if not os.path.isdir(cfg.general.save_dir):
        os.makedirs(cfg.general.save_dir)
        shutil.copy(os.path.join(os.path.dirname(config_file), 'infer_config.py'), cfg.general.save_dir)

    # Update training config file
    shutil.copy(config_file, os.path.join(cfg.general.save_dir, os.path.basename(config_file)))

    # enable logging
    log_file = os.path.join(cfg.general.save_dir, 'train_log.txt')
    logger = file_io.setup_logger(log_file, 'obj_det2d')

    # control randomness during training
    np.random.seed(cfg.debug.seed)
    torch.manual_seed(cfg.debug.seed)
    if cfg.general.num_gpus > 0:
        torch.cuda.manual_seed(cfg.debug.seed)

    # read training and validation label files
    labels_tr = pd.read_csv(cfg.general.train_label_file, na_filter=False)
    data_folder_tr = cfg.general.train_image_folder
    print(f'{len(os.listdir(data_folder_tr))} pictures in {data_folder_tr}')

    labels_dev = pd.read_csv(cfg.general.val_label_file, na_filter=False)
    data_folder_dev = cfg.general.val_image_folder
    print(f'{len(os.listdir(data_folder_dev))} pictures in {data_folder_dev}')

    labels_tr = labels_tr.loc[labels_tr['annotation'].astype(bool)].reset_index(drop=True)
    img_class_dict_tr = dict(zip(labels_tr.image_name, labels_tr.annotation))
    img_class_dict_dev = dict(zip(labels_dev.image_name, labels_dev.annotation))

    # load training dataset
    dataset_train = ObjectDetectionDataset(
        data_folder=data_folder_tr,
        data_type='train',
        labels_dict=img_class_dict_tr,
        resize_size=cfg.dataset.resize_size,
        normalizer=cfg.dataset.normalizer,
        augmentations=cfg.augmentations
    )
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.batch_size,
        collate_fn=train_utils.collate_fn
    )

    # load validation dataset
    dataset_dev = ObjectDetectionDataset(
        data_folder=data_folder_dev,
        data_type='val',
        labels_dict=img_class_dict_dev,
        resize_size=cfg.dataset.resize_size,
        normalizer=cfg.dataset.normalizer
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_dev,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=train_utils.collate_fn
    )

    model_module = importlib.import_module('detection2d.network.' + cfg.net.name)
    model = model_module.get_detection_model(cfg.dataset.num_classes, cfg.net.pre_trained,
                                             image_mean=[0, 0, 0], image_std=[1, 1, 1])

    if cfg.general.num_gpus > 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optmr, schdr = cfg.train.optimizer, cfg.train.scheduler
    if optmr.name == 'SGD':
        optimizer = torch.optim.SGD(
            params, lr=cfg.train.lr, momentum=optmr.momentum, weight_decay=optmr.weight_decay
        )

    elif optmr.name == 'Adam':
        optimizer = torch.optim.Adam(
            params, lr=cfg.train.lr, betas=optmr.betas, weight_decay=optmr.weight_decay
        )

    else:
        raise ValueError('Unsupported optimizer type!')

    auc_max = 0
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=schdr.step_size, gamma=schdr.gamma)
    for epoch in range(cfg.train.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=cfg.train.print_freq)
        lr_scheduler.step(epoch)

        model.eval()
        val_pred = []
        val_label = []
        for batch_i, (image, label, width, height) in enumerate(data_loader_val):
            image = list(img.to(device) for img in image)

            val_label.append(label[-1])

            outputs = model(image)
            if len(outputs[-1]['boxes']) == 0:
                val_pred.append(0)
            else:
                val_pred.append(torch.max(outputs[-1]['scores']).tolist())

        val_pred_label = []
        for i in range(len(val_pred)):
            if val_pred[i] >= 0.5:
                val_pred_label.append(1)
            else:
                val_pred_label.append(0)

        number = 0
        for i in range(len(val_pred_label)):
            if val_pred_label[i] == val_label[i]:
                number += 1
        acc = number / len(val_pred_label)
        auc = roc_auc_score(val_label, val_pred)

        msg = 'Epoch: {} | val acc: {:.4f} | val auc: {:.4f}'.format(epoch, acc, auc)
        logger.info(msg)

        if auc > auc_max:
            auc_max = auc
            msg = 'Best Epoch: {} | val acc: {:.4f} | Best val auc: {:.4f}'.format(epoch, acc, auc_max)
            logger.info(msg)
            torch.save(model.state_dict(), os.path.join(cfg.general.save_dir, 'model.pt'))


def main():

    long_description = "Training engine for 2d medical image object detection"
    parser = argparse.ArgumentParser(description=long_description)

    parser.add_argument('-i', '--input',
                        default='/home/qinliu19/projects/Medical-Detection2d-Toolkit/detection2d/config/train_config.py',
                        help='configure file for medical image segmentation training.')
    parser.add_argument('-g', '--gpus',
                        default=2,
                        help='the device id of gpus.')
    args = parser.parse_args()

    train(args.input, args.gpus)


if __name__ == '__main__':
    main()
