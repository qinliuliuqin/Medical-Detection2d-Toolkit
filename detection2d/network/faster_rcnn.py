import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(num_classes, pretrained):
    """
    Get the detection model
    :param num_classes:
    :return:
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
