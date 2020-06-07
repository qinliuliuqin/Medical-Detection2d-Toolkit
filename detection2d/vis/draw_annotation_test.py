import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image

from detection2d.vis.draw_annotation import draw_annotation


data_dir = '/mnt/projects/CXR_Object/'
OBJECT_SEP = ';'
ANNOTATION_SEP = ' '


train_csv = os.path.join(data_dir, 'train.csv')
labels_tr = pd.read_csv(train_csv, na_filter=False)

# viz
fig, axs = plt.subplots(
    nrows=1, ncols=4, subplot_kw=dict(xticks=[], yticks=[]), figsize=(24, 6)
)

example_idxes = [58, 1850, 2611, 6213]
for row, ax in zip(
        labels_tr.iloc[example_idxes].itertuples(index=False), axs
):
    im_path = data_dir + "train/" + row.image_name
    im = Image.open(im_path).convert("RGB")
    if row.annotation:
        draw_annotation(im, row.annotation)

    ax.imshow(im)
    ax.set_title(f"{row.image_name}")