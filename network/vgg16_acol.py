import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.model_zoo import load_url


__all__ = [
    'VGG', 'vgg16_acol',
]

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, drop_thr=0.7, turnoff=False):
        super(VGG, self).__init__()

        # turnoff classifier B
        self.turnoff = turnoff

        self.features = features
        self.classifier_A = make_classifier(512, num_classes)

        if not turnoff:
            self.classifier_B = make_classifier(512, num_classes)

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        if init_weights:
            self._initialize_weights()
        self.thr_val = drop_thr
        self.print_layers = dict()
        self.score = None
        self.label = None
        self.feat_map_a = None
        self.feat_map_b = None
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(self, x, label=None):

        # predicted logits
        logits = list()

        # Feature Extraction
        feature = self.features(x)

        # F.avg_pool2d?
        # only for smoothing?
        feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)

        # Branch A
        feat_map_a = self.classifier_A(feature)
        self.feat_map_a = feat_map_a
        logit_a = self.GlobalAvgPool(feat_map_a)
        b, c, _, _ = logit_a.size()
        logit_a = logit_a.view(b, c)
        logits.append(logit_a)

        self.score = logit_a
        if label is None:
            _, label = torch.max(logit_a, dim=1)
        self.label = label

        if self.turnoff:
            return logits
        # Getting Attention Map
        attention = self.get_attention(feat_map_a, label)

        # Erasing Step
        erased_feature = self.erase_attention(feature, attention, self.thr_val)

        # Branch B
        feat_map_b = self.classifier_B(erased_feature)
        self.feat_map_b = feat_map_b
        logit_b = self.GlobalAvgPool(feat_map_b)
        b, c, _, _ = logit_b.size()
        logit_b = logit_b.view(b, c)
        logits.append(logit_b)

        return logits

    def get_cam(self):
        """
        getting cam image with size (batch, class, h, w)
        """
        # feature map normalize
        normalized_a = normalize_tensor(self.feat_map_a)

        if self.turnoff:
            return normalized_a, self.score
        normalized_b = normalize_tensor(self.feat_map_b)

        # aggregate
        cam = torch.max(normalized_a, normalized_b)
        return cam, self.score

    def get_layers(self):
        return self.print_layers

    def get_attention(self, feat_map, label, normalize=True):
        """
        :return: return attention size (batch, 1, h, w)
        """
        label = label.long()
        b = feat_map.size(0)

        attention = feat_map.detach().clone().requires_grad_(True)[range(b), label.data, :, :]
        attention = attention.unsqueeze(1)
        if normalize:
            attention = normalize_tensor(attention)
        return attention

    def get_loss(self, logits, gt_labels):
        if len(logits) == 1:
            return self.CrossEntropyLoss(logits[0], gt_labels.long())
        elif len(logits) == 2:
            return self.CrossEntropyLoss(logits[0], gt_labels.long()) + \
                   self.CrossEntropyLoss(logits[1], gt_labels.long())

    def erase_attention(self, feature, attention_map, thr_val):
        b, _, h, w = attention_map.size()
        pos = torch.ge(attention_map, thr_val)
        mask = attention_map.new_ones((b, 1, h, w))
        mask[pos.data] = 0.
        erased_feature = feature * mask
        return erased_feature

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def normalize_tensor(x):

    map_size = x.size()
    aggregated = x.view(map_size[0], map_size[1], -1)
    minimum, _ = torch.min(aggregated, dim=-1, keepdim=True)
    maximum, _ = torch.max(aggregated, dim=-1, keepdim=True)
    normalized = torch.div(aggregated - minimum, maximum - minimum)
    normalized = normalized.view(map_size)

    return normalized


def make_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=False),
        nn.Conv2d(1024, out_planes, kernel_size=1, stride=1, padding=0),
    )


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # MaxPool2d layers in ACoL
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'ACoL': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 512, 'M2'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    kwargs['init_weights'] = True
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch], progress=progress)
        state_dict = remove_layer(state_dict, 'classifier.')
        model.load_state_dict(state_dict, strict=False)
    return model


def remove_layer(state_dict, keyword):
    keys = [key for key in state_dict.keys()]
    for key in keys:
        if keyword in key:
            state_dict.pop(key)
    return state_dict


def vgg16_acol(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'ACoL', False, pretrained, progress, **kwargs)
