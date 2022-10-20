"""
# =============================================================================
# Code for models - Unimodal  + Multimodal 
#
#Tomé Albuquerque
# =============================================================================
"""

from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ce = nn.CrossEntropyLoss()

class Base(nn.Module):
    def __init__(self, pretrained_model, n_outputs=4):
        super().__init__()
        self.n_outputs = n_outputs
        model = getattr(models, pretrained_model)(pretrained=True)
        model.conv1 = nn.Conv2d(155, 64, kernel_size=7, stride=2, padding=3,bias=False)
        model = nn.Sequential(*tuple(model.children())[:-1])
        last_dimension = torch.flatten(model(torch.randn(1, 155, 240, 240))).shape[0]
        self.model = nn.Sequential(
            model,
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(last_dimension, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs)
        )

    def forward(self, x):
        return self.model(x)

    def loss(self, Yhat, Y):
        return ce(Yhat, Y)

    def to_proba(self, Yhat):
        return F.softmax(Yhat, 1)

    def to_classes(self, Phat):
        return Phat.argmax(1)


