import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


from detection2d.utils.normalizer import AdaptiveNormalizer
from detection2d.utils.image_tools import *


image_path = '/mnt/projects/CXR_Object/data/train/00029.jpg'
image = Image.open(image_path).convert("RGB")

image_npy = np.array(image)
normalized_image_npy = AdaptiveNormalizer()(image_npy)

normalized_image_npy = convert_to_neg_or_pos_image(normalized_image_npy)
normalized_image = convert_to_PIL_image(normalized_image_npy)

plt.imshow(normalized_image)
plt.show()