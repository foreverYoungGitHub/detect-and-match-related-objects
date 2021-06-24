import os
import cv2
import torch
import numpy as np

from collections import OrderedDict

from .resnet import ResNet18
from .fpn import SSDFPN, SSDFPNMultiPred
from .box import (
    configure_ratio_scale,
    generate_anchors,
    decode,
    nms,
    set_decode,
    set_nms,
)

fpn_config = {
    "SIZES": [[4.0], [4.0], [4.0]],
    "ASPECT_RATIOS": [[1], [1], [1]],
    "FEATURE_LAYER": [[3, 4, 5], [128, 256, 512]],
    "NUM_CLASSES": 2,
}


def create_fpn():
    """ create the model based on the config files
    Returns:
        torch ssds model with backbone as net
    """
    ratios, scales = configure_ratio_scale(
        len(fpn_config["SIZES"]), fpn_config["ASPECT_RATIOS"], fpn_config["SIZES"]
    )
    number_box = [len(r) * len(s) for r, s in zip(ratios, scales)]
    nets_outputs, extras, head = SSDFPN.add_extras(
        feature_layer=fpn_config["FEATURE_LAYER"],
        mbox=number_box,
        num_classes=fpn_config["NUM_CLASSES"],
    )
    model = SSDFPN(
        backbone=ResNet18(outputs=nets_outputs),
        extras=extras,
        head=head,
        num_classes=fpn_config["NUM_CLASSES"],
    )
    return model


def create_fpn_mp():
    """ create the model based on the config files
    Returns:
        torch ssds model with backbone as net
    """
    ratios, scales = configure_ratio_scale(
        len(fpn_config["SIZES"]), fpn_config["ASPECT_RATIOS"], fpn_config["SIZES"]
    )
    number_box = [len(r) * len(s) for r, s in zip(ratios, scales)]
    nets_outputs, extras, head = SSDFPN.add_extras(
        feature_layer=fpn_config["FEATURE_LAYER"],
        mbox=number_box,
        num_classes=fpn_config["NUM_CLASSES"]//2,
    )
    model = SSDFPNMultiPred(
        backbone=ResNet18(outputs=nets_outputs),
        extras=extras,
        head=head,
        num_classes=fpn_config["NUM_CLASSES"]//2,
    )
    return model


def create_anchors(model, image_size, visualize=False):
    """ current version for generate the anchor, only generate the default anchor for each feature map layers
    Returns:
        anchors: OrderedDict(key=stride, value=default_anchors)
    """
    model.eval()
    with torch.no_grad():
        x = torch.rand(
            (1, 3, image_size[1], image_size[0]),
            device=next(model.parameters()).device,
        )
        conf = model(x)[-1]
        strides = [x.shape[-1] // c.shape[-1] for c in conf]

    ratios, scales = configure_ratio_scale(
        len(strides), fpn_config["ASPECT_RATIOS"], fpn_config["SIZES"]
    )
    anchors = OrderedDict(
        [
            (strides[i], generate_anchors(strides[i], ratios[i], scales[i]))
            for i in range(len(strides))
        ]
    )
    if visualize:
        print("Anchor Boxs (width, height)")
        [
            print("Stride {}: {}".format(k, (v[:, 2:] - v[:, :2] + 1).int().tolist()))
            for k, v in anchors.items()
        ]
    return anchors


def create_decoder(
    conf_threshold=0.01, nms_threshold=0.5, top_n=100, top_n_per_level=300
):
    def decoder(loc, conf, anchors):
        decoded = [
            decode(c, l, stride, conf_threshold, top_n_per_level, anchor)
            for l, c, (stride, anchor) in zip(loc, conf, anchors.items())
        ]
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return nms(*decoded, nms_threshold, top_n, using_diou=False)

    return decoder


def create_set_decoder(
    conf_threshold=0.01, nms_threshold=0.5, top_n=100, top_n_per_level=300
):
    def decoder(loc, sloc, conf, anchors):
        decoded = [
            set_decode(c, l, sl, stride, conf_threshold, top_n_per_level, anchor)
            for l, sl, c, (stride, anchor) in zip(loc, sloc, conf, anchors.items())
        ]
        decoded = [torch.cat(tensors, 1) for tensors in zip(*decoded)]
        return set_nms(*decoded, nms_threshold, top_n, using_diou=False)

    return decoder


class Detector(object):
    def __init__(self, model_name, checkpoint, image_size=(320, 320)):
        if model_name == "fpn":
            self.model = create_fpn()
            self.decoder = create_decoder(conf_threshold=0.2)
        elif model_name == "fpn+mp":
            self.model = create_fpn_mp()
            self.decoder = create_set_decoder(conf_threshold=0.2)
        self.anchors = create_anchors(self.model, image_size)
        self.image_size = image_size
        self.mean = 0
        self.std = 255

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if checkpoint:
            print('Loading initial model weights from {:s}'.format(checkpoint))
            self.resume_checkpoint(checkpoint)
        self.model.eval().to(self.device)

    def __call__(self, image: np.ndarray):
        image = image[None, ...].transpose(0,3,1,2)
        image_tensor = torch.Tensor(image).to(self.device)
        image_tensor = (image_tensor - self.mean) / self.std

        detections = self.decoder(*self.model(image_tensor), self.anchors)
        out_scores, out_boxes, out_classes = (
            d.cpu().detach().numpy()[0] for d in detections
        )

        return out_scores, out_boxes, out_classes

    def resize_infer(self, image: np.ndarray):
        image_rs = cv2.resize(image, self.image_size)
        out_scores, out_boxes, out_classes = self.__call__(image_rs)
        out_boxes[:,::2] *= image.shape[1] / image_rs.shape[1]
        out_boxes[:,1::2] *= image.shape[0] / image_rs.shape[0]
        return out_scores, out_boxes.astype(int), out_classes.astype(int)

    def resume_checkpoint(self, resume_checkpoint):
        if resume_checkpoint == '' or not os.path.isfile(resume_checkpoint):
            print(("=> no checkpoint found at '{}'".format(resume_checkpoint)))
            return False
        print(("=> loading checkpoint '{:s}'".format(resume_checkpoint)))
        checkpoint = torch.load(resume_checkpoint, map_location=torch.device('cpu'))
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        self.model.load_state_dict(checkpoint)
        return self.model