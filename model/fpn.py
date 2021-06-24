import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import copy


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes, out_planes, kernel_size, stride, padding=padding, bias=False
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )


class SharedHead(nn.Sequential):
    def __init__(self, out_planes):
        layers = []
        for _ in range(4):
            layers += [
                ConvBNReLU(256, 256, 3)
            ]  # [nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()]
        layers += [nn.Conv2d(256, out_planes, 3, padding=1)]
        super(SharedHead, self).__init__(*layers)


class SSDSBase(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SSDSBase, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes

    # Initialize class head prior
    def initialize_prior(self, layer):
        pi = 0.01
        b = -math.log((1 - pi) / pi)
        nn.init.constant_(layer.bias, b)
        nn.init.normal_(layer.weight, std=0.01)

    def initialize_head(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, std=0.01)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)

    def initialize_extra(self, layer):
        if isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, val=0)


class SSDFPN(SSDSBase):
    """RetinaNet in Focal Loss for Dense Object Detection
    See: https://arxiv.org/pdf/1708.02002.pdf for more details.
    Compared with the original implementation, change the conv2d 
    in the extra and head to ConvBNReLU to helps the model converage easily
    Not add the bn&relu to transforms cause it is followed by interpolate and element-wise sum

    Args:
        backbone: backbone layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
        num_classes: num of classes 
    """

    def __init__(self, backbone, extras, head, num_classes):
        super(SSDFPN, self).__init__(backbone, num_classes)

        # SSD network
        self.transforms = nn.ModuleList(extras[0])
        self.extras = nn.ModuleList(extras[1])
        self.loc = head[0]
        self.conf = head[1]

        self.initialize()

    def initialize(self):
        self.backbone.initialize()
        self.transforms.apply(self.initialize_extra)
        self.extras.apply(self.initialize_extra)
        self.loc.apply(self.initialize_head)
        self.conf.apply(self.initialize_head)
        self.conf[-1].apply(self.initialize_prior)

    def forward(self, x):
        loc, conf = [list() for _ in range(2)]

        # apply bases layers and cache source layer outputs
        features = self.backbone(x)

        x = features[-1]
        features_len = len(features)
        for i in range(len(features))[::-1]:
            if i != features_len - 1:
                xx = F.interpolate(
                    xx, scale_factor=2, mode="nearest"
                ) + self.transforms[i](features[i])
            else:
                xx = self.transforms[i](features[i])
            features[i] = xx

        for i, v in enumerate(self.extras):
            if i < features_len:
                xx = v(features[i])
            elif i == features_len:
                xx = v(x)
            else:
                xx = v(xx)
            loc.append(self.loc(xx))
            conf.append(self.conf(xx))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(conf)

    @staticmethod
    def add_extras(feature_layer, mbox, num_classes):
        nets_outputs, transform_layers, extra_layers = [list() for _ in range(3)]
        if not all(mbox[i] == mbox[i + 1] for i in range(len(mbox) - 1)):
            raise ValueError(
                "For SSDFPN module, the number of box have to be same in every layer"
            )
        loc_layers = SharedHead(mbox[0] * 4)
        conf_layers = SharedHead(mbox[0] * num_classes)

        for layer, depth in zip(feature_layer[0], feature_layer[1]):
            if isinstance(layer, int):
                nets_outputs.append(layer)
                transform_layers += [
                    nn.Conv2d(depth, 256, 1)
                ]  # [ConvBNReLU(depth, 256, 1)]
                extra_layers += [
                    ConvBNReLU(256, 256, 3)
                ]  # [nn.Conv2d(256, 256, 3, padding=1)]
            elif layer == "Conv:S":
                extra_layers += [
                    ConvBNReLU(depth, 256, 3, stride=2)
                ]  # [nn.Conv2d(depth, 256, 3, stride=2, padding=1)]
            else:
                raise ValueError(layer + " does not support by SSDFPN")
        return nets_outputs, (transform_layers, extra_layers), (loc_layers, conf_layers)


class SSDFPNMultiPred(SSDFPN):
    def __init__(self, backbone, extras, head, num_classes):
        super(SSDFPNMultiPred, self).__init__(backbone, extras, head, num_classes)
        self.sloc = copy.deepcopy(head[0])
        self.sloc.apply(self.initialize_head)

    def forward(self, x):
        loc, sloc, conf = [list() for _ in range(3)]

        # apply bases layers and cache source layer outputs
        features = self.backbone(x)

        x = features[-1]
        features_len = len(features)
        for i in range(len(features))[::-1]:
            if i != features_len - 1:
                xx = F.interpolate(xx, scale_factor=2) + self.transforms[i](features[i])
            else:
                xx = self.transforms[i](features[i])
            features[i] = xx

        for i, v in enumerate(self.extras):
            if i < features_len:
                xx = v(features[i])
            elif i == features_len:
                xx = v(x)
            else:
                xx = v(xx)
            loc.append(self.loc(xx))
            sloc.append(self.sloc(xx))
            conf.append(self.conf(xx))

        if not self.training:
            conf = [c.sigmoid() for c in conf]
        return tuple(loc), tuple(sloc), tuple(conf)
