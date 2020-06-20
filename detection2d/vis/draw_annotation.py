from PIL import ImageDraw


def draw_annotation(im, anno_str, obj_sep=';', annot_sep=' ', fill=(255, 63, 63, 40)):
    """
    Draw annotations for the input image
    :param im:
    :param anno_str:
    :param fill:
    :return:
    """
    draw = ImageDraw.Draw(im, mode="RGBA")
    for anno in anno_str.split(obj_sep):
        anno = list(map(int, anno.split(annot_sep)))
        if anno[0] == 0:
            draw.rectangle(anno[1:], fill=fill)
        elif anno[0] == 1:
            draw.ellipse(anno[1:], fill=fill)
        else:
            draw.polygon(anno[1:], fill=fill)


def draw_rectangle(im, boxes, fill=(255, 63, 63, 40)):
    """
    Draw the annotated rectange bounding box for the input image
    :param im:
    :param boxes:
    :return:
    """
    draw = ImageDraw.Draw(im, mode='RGBA')
    for box in boxes:
        draw.rectangle(box, fill=fill)