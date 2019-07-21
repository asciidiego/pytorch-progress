import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        # [input image channels, output image channels, kernel size]
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # An affine operation: y = W * x + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6 * 6 is the image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 is the number of classes
    
    def forward(self, x):
        # Max-pooling over a two by two window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, size can be input as a scalar
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(f"# flat features: {self.num_flat_features(x)}")
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # All except the batch size dimension
        size = x.size()[1:]
        num_features = 1
        for dim in size:
            num_features *= dim
        return num_features

net = Net()
print(net)

# TODO: Backpropagation