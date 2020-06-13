import argparse
from collections import namedtuple
import numpy as np
from skimage.measure import points_in_poly

Object = namedtuple('Object', ['image_path', 'object_id', 'object_type', 'coordinates'])
Prediction = namedtuple('Prediction', ['image_path', 'probability', 'coordinates'])


def inside_object(pred, obj):
    """
    Test whether the predicted bounding box is inside the ground truth bounding box.
    :param pred:
    :param obj:
    :return: bool
    """
    # bounding box
    if obj.object_type == '0':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        return x1 <= x <= x2 and y1 <= y <= y2
    # bounding ellipse
    if obj.object_type == '1':
        x1, y1, x2, y2 = obj.coordinates
        x, y = pred.coordinates
        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
        x_axis, y_axis = (x2 - x1) / 2, (y2 - y1) / 2
        return ((x - x_center)/x_axis)**2 + ((y - y_center)/y_axis)**2 <= 1
    # mask/polygon
    if obj.object_type == '2':
        num_points = len(obj.coordinates) // 2
        poly_points = obj.coordinates.reshape(num_points, 2, order='C')
        return points_in_poly(pred.coordinates.reshape(1, 2), poly_points)[0]


def evaluate(gt_csv_path, pred_csv_path, fps):
    """
    Evaluate the localization error
    :param gt_csv_path:
    :param pred_csv_path:
    :param fps:
    :return:
    """
    # parse ground truth csv
    num_image = 0
    num_object = 0
    object_dict = {}
    with open(gt_csv_path) as f:
        # header
        next(f)
        for line in f:
            image_path, annotation = line.strip('\n').split(',')

            if annotation == '':
                num_image += 1
                continue

            object_annos = annotation.split(';')
            for object_anno in object_annos:
                fields = object_anno.split(' ')
                object_type = fields[0]
                coords = np.array(list(map(float, fields[1:])))
                obj = Object(image_path, num_object, object_type, coords)
                if image_path in object_dict:
                    object_dict[image_path].append(obj)
                else:
                    object_dict[image_path] = [obj]
                num_object += 1
            num_image += 1

    # parse prediction truth csv
    preds = []
    with open(pred_csv_path) as f:
        # header
        next(f)
        for line in f:
            image_path, prediction = line.strip('\n').split(',')

            if prediction == '':
                continue

            coord_predictions = prediction.split(';')
            for coord_prediction in coord_predictions:
                fields = coord_prediction.split(' ')
                probability, x, y = list(map(float, fields))
                pred = Prediction(image_path, probability, np.array([x, y]))
                preds.append(pred)

    # sort prediction by probability
    preds = sorted(preds, key=lambda x: x.probability, reverse=True)

    # compute hits and false positives
    hits = 0
    false_positives = 0
    fps_idx = 0
    object_hitted = set()
    fps = list(map(float, fps.split(',')))
    froc = []
    for i in range(len(preds)):
        is_inside = False
        pred = preds[i]
        if pred.image_path in object_dict:
            for obj in object_dict[pred.image_path]:
                if inside_object(pred, obj):
                    is_inside = True
                    if obj.object_id not in object_hitted:
                        hits += 1
                        object_hitted.add(obj.object_id)

        if not is_inside:
            false_positives += 1

        if false_positives / num_image >= fps[fps_idx]:
            sensitivity = hits / num_object
            froc.append(sensitivity)
            fps_idx += 1

            if len(fps) == len(froc):
                break

    while len(froc) < len(fps):
        froc.append(froc[-1])

    # print froc
    print('False positives per image:')
    print(fps)
    print('Sensitivity:')
    print('\t'.join(map(lambda x: '{:.3f}'.format(x), froc)))
    print('FROC:')
    print(np.mean(froc))



def main():

    parser = argparse.ArgumentParser(description='Compute FROC')
    parser.add_argument('-g', '--gt-csv',
                        default='/shenlab/lab_stor6/projects/CXR_Object/dev.csv',
                        metavar='GT_CSV',
                        help="Path to the ground truch csv file")
    parser.add_argument('-p', '--pred-csv',
                        default='/shenlab/lab_stor6/qinliu/CXR_Object/results/model_0610_2020/localization.csv',
                        metavar='PRED_PATH',
                        help="Path to the predicted csv file")
    parser.add_argument('-f', '--fps',
                        default='0.125,0.25,0.5,1,2,4,8',
                        help='False positives per image to compute FROC, comma seperated')

    args = parser.parse_args()

    evaluate(args.gt_csv, args.pred_csv, args.fps)


if __name__ == '__main__':

    main()
