"""
ASL Hand Sign Digit Classification — Model Definition
AI 100 — Final Project
Authors: Jasraj "Jay" Raval & Jack Sweeney
"""
import torch
import torch.nn as nn


class ASL_CNN(nn.Module):
    """
    Custom CNN for ASL digit classification (0-9).
    3 Conv Blocks + Fully Connected Classifier.
    Input: [B, 3, 64, 64] normalized RGB image.
    Output: [B, 10] raw logits.
    """
    def __init__(self, num_classes: int = 10):
        super(ASL_CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def forward(self, x):
        return self.classifier(self.features(x))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight); nn.init.zeros_(m.bias)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
