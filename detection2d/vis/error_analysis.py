from collections import namedtuple

from detection2d.utils.bbox_utils import read_boxes_from_annotation_txt, read_boxes_center_from_localization_txt


"""
The struct that contains the error summary.
"""
ErrorSummary = namedtuple('ErrorSummary', 'image_list label pred error')


def error_analysis(image_list, labeled_objects, detected_objects, decending=True):
    """
    Analyze landmark detection error and return the error statistics summary.
    Input arguments:
    label_landmark: A dict whose keys and values are filenames and coordinates of labelled points respectively.
    detection_landmark: A dict whose keys and values are filenames and coordinates of detected points respectively.
    descending:          Flag indicating whether errors sorted in ascending or descending order.
    Return:
    error_summary:       Summary of error statistics.
    """

    # label, pred, error = [], [], []
    # label_names = set(labeled_objects.keys())
    # pred_names = set(detected_objects.keys())
    # for image_name in image_list:
    #     if (image_name in label_names) and not (image_name in pred_names):
    #         label_prob, pred_prob, pred_error = 1.0, 0.0, 1.0
    #
    #     elif not (image_name in label_names) and (image_name in pred_names):
    #         label_prob, pred_prob, pred_error = 0.0, 1.0, 1.0
    #
    #     elif not (image_name in label_names) and not (image_name in pred_names):
    #         label_prob, pred_prob, pred_error = 0.0, 0.0, 0.0
    #
    #     else:
    #         label_prob = 1.0
    #         pred_prob = float(max(read_boxes_center_from_localization_txt(detected_objects[image_name])[1]))
    #         pred_error = abs(label_prob - pred_prob)
    #
    #     label.append(label_prob)
    #     pred.append(pred_prob)
    #     error.append(pred_error)
    #
    # if decending:
    #     tuples = [(error[idx], image_list[idx], label[idx], pred[idx]) for idx in range(len(image_list))]
    #     tuples = sorted(tuples, key=lambda x: x[0], reverse=True)
    #     error = [round(elm[0], 3) for elm in tuples]
    #     image_list = [elm[1] for elm in tuples]
    #     label = [elm[2] for elm in tuples]
    #     pred = [round(elm[3], 3) for elm in tuples]

    error_summary = ErrorSummary(
        image_list=image_list,
        label=[0] * len(image_list),
        pred=[0] * len(image_list),
        error=[0] * len(image_list)
    )

    return error_summary