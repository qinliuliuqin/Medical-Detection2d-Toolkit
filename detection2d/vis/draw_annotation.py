from PIL import ImageDraw


def draw_annotation(im, anno_str, obj_sep=';', annot_sep=' ', fill=(255, 63, 63, 40)):
    """
    Draw annotations for
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