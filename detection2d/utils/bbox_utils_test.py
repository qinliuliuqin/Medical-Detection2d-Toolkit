import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt, draw_rect


img_path = '/mnt/projects/CXR_Object/data/dev/08001.jpg'
img = Image.open(img_path).convert("RGB")

annotation = '0 128 128 512 512'
bboxes, labels = read_boxes_from_annotation_txt(annotation)

color_green = [25, 255, 25]
img = draw_rect(np.array(img), np.array(bboxes), color=color_green)

plt.figure()
plt.imshow(img)
plt.show()
