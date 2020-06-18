# Medical-Detection2d-Toolkit
There are 3 box types: \
0: the conventional bounding box consists of four coordinates 'x1 y1 x2 y2'. \
1: the masked bounding box consists of a consequence of coordinates 'x1 y1 x2 y2 ...'. \
2: the elipse bouding box consists of the center and the radius. \

The class of the annotation box should be in the range of [1, mum_clsses].

The annotation file should be a CSV file in the following format:
```buildoutcfg
,image_name,annotation
0,image_name_0,box_type_0 box_cls_type_0 x1 y1 x2 y2; x3 y3 x4 y4
1,image_name_1,box_type_1 box_cls_type_1 x5 y5 x6 y6; x7 y7 x8 y8
...
```
