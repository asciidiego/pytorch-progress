import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # [input image channels, output image channels, kernel size]
        self.conv1 = nn.Conv2D(1, 6, 3)
        self.conv2 = nn.Conv2D(6, 16, 3)
        # An affine operation: y = W * x + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6 * 6 is the image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 is the number of classes
