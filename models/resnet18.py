import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)

class Classifier(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(Classifier, self).__init__()

        self.in_planes = in_planes
        self.num_classes = num_classes

        layers = [nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()]

        self.layers = nn.Sequential(*layers)
        self.ee_fc = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        output = self.layers(x)
        return self.ee_fc(output)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # x -> [64, 32, 32]
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # x -> [128, 16, 16]
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # x -> [256, 8, 8]
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # x -> [512, 4, 4]
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.ee1 = Classifier(64, num_classes)
        self.ee2 = Classifier(128, num_classes)
        self.ee3 = Classifier(256, num_classes)
        self.ee4 = Classifier(512, num_classes)

        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.ee_classifiers = [self.ee1, self.ee2, self.ee3, self.ee4]


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feat=False):
        outs = []
        x = F.relu(self.bn1(self.conv1(x)))

        for layer, ee in zip(self.layers, self.ee_classifiers):
            x = layer(x)
            outs.append(ee(x))

        if return_feat:
            feat_pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_flatten = nn.Flatten()
            return outs, feat_flatten(feat_pool(x))
        return outs

def resnet18(args, params):
    return ResNet(block=BasicBlock,
                  num_blocks=[2, 2, 2, 2],
                  num_classes=args.class_num)