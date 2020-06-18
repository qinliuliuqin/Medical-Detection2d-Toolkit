import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

from detection2d.vis.draw_annotation import draw_annotation


data_dir = '/mnt/projects/CXR_Object'
OBJECT_SEP = ';'
ANNOTATION_SEP = ' '

save_fig_folder = '/mnt/projects/CXR_Object/results/vis'
if not os.path.isdir(save_fig_folder):
    os.makedirs(save_fig_folder)

train_csv = os.path.join(data_dir, 'dataset', 'train.csv')
labels_tr = pd.read_csv(train_csv, na_filter=False)

# viz
for idx in range(len(labels_tr)):
    print(idx)
    image_name = labels_tr.iloc[idx]['image_name']
    annotation = labels_tr.iloc[idx]['annotation']

    image_path = os.path.join(data_dir, "data", "train", image_name)
    image = Image.open(image_path).convert("RGB")
    if annotation:
        draw_annotation(image, annotation)

    plt.imshow(image)
    plt.title('{}'.format(image_name))
    plt.savefig(os.path.join(save_fig_folder, image_name))