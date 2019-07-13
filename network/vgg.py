import torch
import torch.nn as nn
import os
from torch.utils.model_zoo import load_url
from torchvision.utils import make_grid, save_image


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, writer=None, args=None):
        super(VGG, self).__init__()
        self.features = features

        self.classifier_A = make_classifier(512, num_classes)
        self.classifier_B = make_classifier(512, num_classes)

        self.GlobalAvgPool = nn.AdaptiveAvgPool2d(1)
        if init_weights:
            self._initialize_weights()
        self.writer = writer
        self.args = args
        self.iter = 0
        self.iter2 = 0
    def forward(self, x, label=None, thr_val=0.7):
        self.iter += 1
        # Backbone
        feature = self.features(x)

        # F.avg_pool2d? why?

        # Branch A
        feature_map_A = self.classifier_A(feature)
        self.feature_map_A = feature_map_A
        logits_A = self.GlobalAvgPool(feature_map_A)
        b, c, _, _ = logits_A.size()
        logits_A = logits_A.view(b, c)

        if label is None:
            print('label is none')
            _, label = torch.max(logits_A, dim=1)

        # generate attention map
        attention_map = self.get_attention_map(feature_map_A, label)
        # erasing step
        erased_feature = self.erase_attention(feature, attention_map, thr_val)

        # Branch B
        feature_map_B = self.classifier_B(erased_feature)
        self.feature_map_B = feature_map_B
        logits_B = self.GlobalAvgPool(feature_map_B)
        b, c, _, _ = logits_B.size()
        logits_B = logits_B.view(b, c)

        if self.training:
            if self.iter % 90 == 0 and self.args.gpu == 0 and self.args.image_save:
                file_path = os.path.join('image_path', self.args.name)
                if not os.path.isdir(file_path):
                    os.mkdir(file_path)
                file_name1 = os.path.join(file_path, str(self.iter)+'_attention.jpg')

                save_image2 = torch.mean(erased_feature.detach().cpu(), dim=1, keepdim=True)
                concat = torch.cat((attention_map.cpu().unsqueeze(1), save_image2), dim=3)

                save_image(concat, file_name1)

        return logits_A, logits_B

    def get_attention_map(self, feature_map, label, normalize=True):
        label = label.long()
        b = feature_map.size(0)

        attention_map = feature_map[range(b),label,:,:]

        if normalize:
            attention_map = self.normalize_attention(attention_map)

        return attention_map

    def erase_attention(self, feature, attention_map, thr_val):

        b, h, w = attention_map.size()
        pos = torch.ge(attention_map, thr_val)
        mask = attention_map.new_ones((b,h,w))
        mask[pos.data] = 0.
        mask = torch.unsqueeze(mask, dim=1)

        if self.training:
            save_image2 = make_grid(mask.detach())
            self.writer.add_image(self.args.name + '/erased_map', save_image2, 0)
        # erase feature
        erased_feature = feature * mask
        return erased_feature

    def normalize_attention(self, attention_map):
        map_size = attention_map.size()

        aggregated_attention_map = attention_map.view(map_size[0], map_size[1],-1)
        minimum, _ = torch.min(aggregated_attention_map, dim=-1, keepdim=True)
        maximum, _ = torch.max(aggregated_attention_map, dim=-1, keepdim=True)
        normalized_attention_map = torch.div(aggregated_attention_map - minimum,
                                             maximum - minimum)
        normalized_attention_map = normalized_attention_map.view(map_size)

        return normalized_attention_map

    def generate_localization_map(self, x, label=None, thr_val=0.5):
        self.iter2 += 1
        # Backbone
        feature = self.features(x)
        # F.avg_pool2d? why?

        # Branch A
        feature_map_A = self.classifier_A(feature)
        logits_A = self.GlobalAvgPool(feature_map_A)
        b, c, _, _ = logits_A.size()
        logits_A = logits_A.view(b, c)

        if label is None:
            _, label = torch.max(logits_A, dim=1)

        # generate attention map
        attention_map = self.get_attention_map(feature_map_A, label)
        # erasing step
        erased_feature = self.erase_attention(feature, attention_map, thr_val)

        # Branch B
        feature_map_B = self.classifier_B(erased_feature)

        feature_map_A = self.get_attention_map(feature_map_A, label).unsqueeze(1)
        feature_map_B = self.get_attention_map(feature_map_B, label).unsqueeze(1)
        map_A = self.normalize_attention(feature_map_A)
        map_B = self.normalize_attention(feature_map_B)
        aggregated_map = torch.max(map_A, map_B).detach().cpu()

        if self.iter2 % 90 == 0 and self.args.gpu == 0 and self.args.image_save:

            compare_img = torch.cat((map_A.detach().cpu(), map_B.detach().cpu(), aggregated_map), dim=3)

            file_path = os.path.join('image_path', self.args.name)
            if not os.path.isdir(file_path):
                os.mkdir(file_path)
            file_name1 = os.path.join(file_path, str(self.iter) + '_concat.jpg')
            save_image(compare_img, file_name1, normalize=True, nrow=4)

        upsampled_map = torch.nn.functional.interpolate(aggregated_map,
                                                        size=(224,224),
                                                        mode='bilinear',
                                                        align_corners=True)

        return upsampled_map

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_classifier(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
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
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'ACoL' : [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512, 512, 'M2', 512, 512, 512, 'M2'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, writer, args, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), writer=writer, args=args, **kwargs)
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


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, writer=None, args=None, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'ACoL', False, pretrained, progress, writer, args, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>'_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)
