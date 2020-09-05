import matplotlib.pyplot as plt
import numpy as np
import os
import pydicom
import PIL.Image as Image

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt, draw_rect


# data_dir = '/mnt/projects/CXR_Object'
# OBJECT_SEP = ';'
# ANNOTATION_SEP = ' '
#
# save_fig_folder = '/mnt/projects/CXR_Object/results/vis'
# if not os.path.isdir(save_fig_folder):
#     os.makedirs(save_fig_folder)
#
# train_csv = os.path.join(data_dir, 'dataset', 'train.csv')
# labels_tr = pd.read_csv(train_csv, na_filter=False)
#
# # viz
# for idx in range(len(labels_tr)):
#     print(idx)
#     image_name = labels_tr.iloc[idx]['image_name']
#     annotation = labels_tr.iloc[idx]['annotation']
#
#     image_path = os.path.join(data_dir, "data", "train", image_name)
#     image = Image.open(image_path).convert("RGB")
#     if annotation:
#         draw_annotation(image, annotation)
#
#     plt.imshow(image)
#     plt.title('{}'.format(image_name))
#     plt.savefig(os.path.join(save_fig_folder, image_name))

def gen_plane_images(error_summary, image_folder, labels_dict, preds_dict, error_threshold, output_folder):
    """

    :param image_list:
    :param image_folder:
    :param labels_dict:
    :param preds_dict:
    :param error_summary:
    :param error_threshold:
    :param output_folder:
    :return:
    """

    output_picture_folder = os.path.join(output_folder, 'pictures')
    if not os.path.isdir(output_picture_folder):
        os.makedirs(output_picture_folder)

    image_list = error_summary.image_list
    for image_idx, image_name in enumerate(image_list):
        print(image_name)
        if image_name.endswith('.dcm'):
            img_dcm = pydicom.read_file(os.path.join(image_folder, image_name))
            img_npy = np.expand_dims(img_dcm.pixel_array, axis=2)
            img_npy = np.concatenate([img_npy, img_npy, img_npy], axis=2)
            image = Image.fromarray(img_npy)

        else:
            image = Image.open(os.path.join(image_folder, image_name)).convert('RGB')

        image_pre, image_post = image_name.split('.')
        if image_post == 'dcm':
            image_post = 'jpg'

        label_image_name = '{}_labelled.{}'.format(image_pre, image_post)

        image_npy = np.array(image)

        if image_name in labels_dict.keys():
            annotation = labels_dict[image_name]
            bboxes, _ = read_boxes_from_annotation_txt(annotation)
            image_npy = draw_rect(np.array(image_npy), np.array(bboxes), color=[25, 255, 25])



        if preds_dict is not None and image_name in preds_dict:
            if image_name in preds_dict.keys():
                annotation = preds_dict[image_name]
                bboxes, _ = read_boxes_from_annotation_txt(annotation)
                image_npy = draw_rect(np.array(image_npy), np.array(bboxes), color=[255, 25, 25])

        if image_name in labels_dict.keys() and preds_dict is not None and image_name in preds_dict:
            # Create a new figure
            fig = plt.figure(1, figsize=(5, 5))
            plt.imshow(image_npy)

            # Save and close the figure.
            fig.savefig(os.path.join(output_picture_folder, label_image_name))
            fig.clf()
