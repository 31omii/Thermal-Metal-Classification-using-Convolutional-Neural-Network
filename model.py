import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
)

from config import NUM_CLASSES


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        hidden = max(channels // reduction, 4)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):

        b, c, _, _ = x.size()

        y = self.pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.se = SEBlock(out_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.shortcut = nn.Identity()

    def forward(self, x):

        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        out = self.relu(out)
        out = self.pool(out)

        return out


class CustomCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Feature Extractor
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64)
        self.layer3 = ResidualBlock(64, 128)
        self.layer4 = ResidualBlock(128, 256)

        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0)


class ResNET18(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None

        self.backbone = models.resnet18(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


class ResNET50(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None

        self.backbone = models.resnet50(weights=weights)

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


class DenseNET_121(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        weights = DenseNet121_Weights.DEFAULT if pretrained else None

        self.backbone = models.densenet121(weights=weights)

        in_features = self.backbone.classifier.in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


class EfficientNET_B0(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None

        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


class EfficientNET_B3(nn.Module):
    def __init__(
        self,
        num_classes=NUM_CLASSES,
        pretrained=True,
        dropout=0.5,
    ):
        super().__init__()

        weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None

        self.backbone = models.efficientnet_b3(weights=weights)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        return self.backbone(x)


def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def model_summary(model):

    total, trainable = count_parameters(model)
    print()
    print(f"Model             : {model.__class__.__name__}")
    print(f"Parameters        : {total:,}")
    print(f"Trainable Params  : {trainable:,}")
    print(f"Model Size        : {total * 4 / 1024**2:.2f} MB")


# Model factory
def get_model(model_name):

    if model_name == "custom_cnn":
        return CustomCNN()

    elif model_name == "resnet18":
        return ResNET18()

    elif model_name == "resnet50":
        return ResNET50()

    elif model_name == "densenet121":
        return DenseNET_121()

    elif model_name == "efficientnet_b0":
        return EfficientNET_B0()

    elif model_name == "efficientnet_b3":
        return EfficientNET_B3()

    raise ValueError(f"Unknown model : {model_name}")