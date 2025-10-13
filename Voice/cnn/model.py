import torch
import torch.nn as nn
import torchvision.models as models


class CNNModel(nn.Module):
    """ResNet-50 기반 CNN 모델"""

    def __init__(self, num_classes=2):
        super(CNNModel, self).__init__()

        # ImageNet 사전학습된 ResNet-50 로드
        self.resnet = models.resnet50(pretrained=True)

        # 첫 번째 Conv 레이어 수정 (grayscale 입력용)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 마지막 FC 레이어 수정 (2-class 분류)
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet(x)
