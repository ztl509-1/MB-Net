import torch
import torch.nn as nn
import os
import math
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary

class MG_Net(nn.Module):
    def __init__(self):
        super(MG_Net, self).__init__()
        channel = 32
        kernel = 5
        self.pool = 16

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(644, 1,bias=False),
            nn.Sigmoid(),
        )
        self.group3 = nn.Sequential(
            nn.MaxPool1d(2,  stride=2),
            nn.Conv1d(2, 1, kernel_size=kernel, stride=1),
            nn.BatchNorm1d(1),
            nn.ReLU(),
            #nn.AdaptiveAvgPool1d(4),
            nn.AvgPool1d(299,299)
        )
        self.group2 = nn.Sequential(
            nn.MaxPool1d(2, stride=2),
            nn.Conv1d(2, channel//2, kernel_size=kernel, stride=1),
            nn.BatchNorm1d(channel//2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(channel//2, channel//2, kernel_size=kernel, stride=1, padding=0, groups=channel//2),
            nn.BatchNorm1d(channel//2),
            nn.ReLU(),

            #nn.AdaptiveAvgPool1d(8),
            nn.AvgPool1d(36,36)
        )
        self.group1 = nn.Sequential(
            nn.Conv1d(2, channel, kernel_size=kernel, stride=1),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(channel, channel, kernel_size=kernel, stride=1, padding=0, groups=channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(channel, channel, kernel_size=kernel, stride=1, padding=0, groups=channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            #nn.AdaptiveAvgPool1d(16),
            nn.AvgPool1d(18,18),
        )

    def forward(self, x):
        x1 = x[:, :, 0:1200]
        x2 = x[:, :, 1200:2400]
        x3 = x[:, :, 2400:4800]

        x11 = self.group1(x1)
        x22 = self.group2(x2)
        x33 = self.group3(x3)

        x1 = x11.reshape(-1, 16 * 32)
        x2 = x22.reshape(-1, 8 * 32//2)
        x3 = x33.reshape(-1, 4 * 1)

        x = torch.cat((x1, x2, x3), dim=1)
        out = self.fc(x)

        return x11, x22, x33, out

if __name__ == "__main__":
    x = torch.rand(2, 2, 4800)
    model = MG_Net()
    y = model(x)
    summary(model.cuda(), (2,4800))