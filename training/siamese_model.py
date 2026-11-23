import torch
import torch.nn as nn
from torchvision import models

class SiameseDamageClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, num_classes),
        )

    def forward(self, before, after):
        f1 = self.encoder(before).flatten(1)
        f2 = self.encoder(after).flatten(1)
        difference = torch.abs(f1 - f2)
        return self.classifier(difference)
