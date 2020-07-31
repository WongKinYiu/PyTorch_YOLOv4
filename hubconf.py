"""File for accessing YOLOv4 via PyTorch Hub https://pytorch.org/hub/

Usage:
    import torch
    model = torch.hub.load('AlexeyAB/PyTorch_YOLOv4:u5_preview', 'yolov4_pacsp_s', pretrained=True, channels=3, classes=80)
"""

dependencies = ['torch', 'yaml']

import os

import torch

from models.yolo import Model
from utils import google_utils
import urllib.request


def create(name, pretrained, channels, classes):
    """Creates a specified YOLOv4 model

    Arguments:
        name (str): name of model, i.e. 'yolov4_pacsp_s'
        pretrained (bool): load pretrained weights into the model
        channels (int): number of input channels
        classes (int): number of model classes

    Returns:
        pytorch model
    """
    config = os.path.join(os.path.dirname(__file__), 'models', '%s.yaml' % name)  # model.yaml path
    try:
        model = Model(config, channels, classes)
        if pretrained:
            ckpt = '%s.pt' % name  # checkpoint filename
            url_name = 'https://github.com/AlexeyAB/PyTorch_YOLOv4/releases/download/models_29_07_2020_u5/' + ckpt
            print(url_name)
            dst_ckpt = './' + ckpt
            print('dst_ckpt:', dst_ckpt)
            urllib.request.urlretrieve(url_name, dst_ckpt)
            # google_utils.attempt_download(ckpt)  # download if not found locally
            state_dict = torch.load(dst_ckpt, map_location=torch.device('cpu'))['model'].float().state_dict()  # to FP32
            state_dict = {k: v for k, v in state_dict.items() if model.state_dict()[k].shape == v.shape}  # filter
            model.load_state_dict(state_dict, strict=False)  # load
        return model

    except Exception as e:
        s = 'Cache maybe be out of date, deleting cache and retrying may solve this. See %s for help.' % help_url
        raise Exception(s) from e


def yolov4_pacsp_s(pretrained=False, channels=3, classes=80):
    """YOLOv4-small model from https://github.com/AlexeyAB/PyTorch_YOLOv4/releases/tag/models_29_07_2020

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov4-pacsp-s', pretrained, channels, classes)


def yolov4_pacsp(pretrained=False, channels=3, classes=80):
    """YOLOv4-medium model from https://github.com/AlexeyAB/PyTorch_YOLOv4/releases/tag/models_29_07_2020

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov4-pacsp', pretrained, channels, classes)


def yolov4_pacsp_x(pretrained=False, channels=3, classes=80):
    """YOLOv4-large model from https://github.com/AlexeyAB/PyTorch_YOLOv4/releases/tag/models_29_07_2020

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov4-pacsp-x', pretrained, channels, classes)


def yolov4_tiny(pretrained=False, channels=3, classes=80):
    """YOLOv4-xlarge model from https://github.com/AlexeyAB/PyTorch_YOLOv4/releases/tag/models_29_07_2020

    Arguments:
        pretrained (bool): load pretrained weights into the model, default=False
        channels (int): number of input channels, default=3
        classes (int): number of model classes, default=80

    Returns:
        pytorch model
    """
    return create('yolov4-tiny', pretrained, channels, classes)
