import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image
import pydicom

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt, draw_rect


def test_picture():
    img_path = '/mnt/projects/CXR_Object/data/dev/08001.jpg'
    img = Image.open(img_path).convert("RGB")

    annotation = '0 128 128 512 512'
    bboxes, labels = read_boxes_from_annotation_txt(annotation)

    color_green = [25, 255, 25]
    img = draw_rect(np.array(img), np.array(bboxes), color=color_green)

    plt.figure()
    plt.imshow(img)
    plt.show()

def test_dicom():
    img_path = '/mnt/projects/CXR_Pneumonia/Stage2/00436515-870c-4b36-a041-de91049b9ab4.dcm'
    img_dcm = pydicom.read_file(img_path)
    img = Image.fromarray(img_dcm.pixel_array)

    # annotation = '0 264 152 477 531'
    annotation='0 562 152 818 605'
    bboxes, labels = read_boxes_from_annotation_txt(annotation)

    color_green = [25, 255, 25]
    img = draw_rect(np.array(img), np.array(bboxes), color=color_green)

    plt.figure()
    plt.imshow(img)
    plt.show()


test_dicom()