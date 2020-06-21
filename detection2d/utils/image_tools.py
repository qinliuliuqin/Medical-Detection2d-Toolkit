import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms


def convert_to_PIL_image(image_npy, mode=None):
    """
    Convert numpy to PIL.Image
    :param image_npy: the input numpy
    :return:
        image (PIL.Image): the output PIL.Image object
    """
    image_npy = image_npy.astype(dtype=np.float32)
    image_npy_uint8 = image_npy.astype(dtype=np.uint8)
    for idx in range(3):
        plane = image_npy[:, :, idx]
        min_val, max_val = np.min(plane), np.max(plane)
        plane = np.array((plane - min_val) / (max_val - min_val) * 255)
        image_npy_uint8[:, :, idx] = plane.astype(dtype=np.uint8)

    return Image.fromarray(image_npy_uint8, mode=mode)


def convert_to_neg_or_pos_image(image_npy, in_place=False):
    """
    convert the negative image to the positive and vise versa.
    :param image_npy:
    :return:
        image (numpy.ndarray): the converted numpy
    """
    if not in_place:
        image_npy = np.copy(image_npy)

    for idx in range(3):
        max_val = np.max(image_npy[:, :, idx])
        image_npy[:, :, idx] = max_val - image_npy[:, :, idx]

    return image_npy