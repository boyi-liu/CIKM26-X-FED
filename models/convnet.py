import torch.nn as nn
import torch.nn.functional as F


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

class DeepConvNet(nn.Module):
    def __init__(self, num_classes):
        super(DeepConvNet, self).__init__()
        hidden_size = 128
        self.conv1 = nn.Conv2d(3, hidden_size, kernel_size=3)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3)
        # self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.ee1 = Classifier(hidden_size, num_classes)
        self.ee2 = Classifier(hidden_size, num_classes)
        self.ee3 = Classifier(hidden_size, num_classes)
        # self.ee4 = Classifier(128, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.layers = [self.conv1, self.conv2, self.conv3, self.conv4]
        # self.ee_classifiers = [self.ee1, self.ee2, self.ee3, self.ee4]
        self.layers = [self.conv1, self.conv2, self.conv3]
        self.ee_classifiers = [self.ee1, self.ee2, self.ee3]


    def forward(self, x, return_feat=False):
        outs = []
        for layer, ee in zip(self.layers, self.ee_classifiers):
            x = F.relu(layer(x))
            x = self.pool(x)
            outs.append(ee(x))

        if return_feat:
            feat_pool = nn.AdaptiveAvgPool2d((1, 1))
            feat_flatten = nn.Flatten()
            return outs, feat_flatten(feat_pool(x))
        return outs

def convnet(args, params):
    return DeepConvNet(num_classes=args.class_num)