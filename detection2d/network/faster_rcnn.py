import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(
        num_classes,
        pretrained,
        # transform parameters
        min_size=800, max_size=1333,
        image_mean=None, image_std=None
):
    """
    Get the detection model
    :param num_classes:
    :return:
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        pretrained_backbone=False,
        # transform parameters
        min_size=min_size, max_size=max_size,
        image_mean=image_mean, image_std=image_std
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
