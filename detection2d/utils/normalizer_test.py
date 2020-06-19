import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from detection2d.utils.normalizer import AdaptiveNormalizer
from detection2d.utils.image_tools import convert_numpy_to_PIL, convert_between_neg_and_pos


image_path = '/mnt/projects/CXR_Object/data/train/00029.jpg'
image = Image.open(image_path).convert("RGB")

image_npy = np.array(image)
normalized_image_npy = AdaptiveNormalizer()(image_npy)

normalized_image_npy = convert_between_neg_and_pos(normalized_image_npy)
normalized_image = convert_numpy_to_PIL(normalized_image_npy)
plt.imshow(normalized_image)
plt.show()