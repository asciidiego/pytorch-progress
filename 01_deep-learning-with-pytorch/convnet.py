import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

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
        print(f"Doing forward pass.")
        print(f"{x.size()}")
        # Max-pooling over a two by two window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, size can be input as a scalar
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        print(f"{x.size()}")
        print(f"Number of flat features: {self.num_flat_features(x)}")
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

# Parameters
net_parameters = list(net.parameters())
print(f"Number of parameters of the net: {len(net_parameters)} ")

# Forward pass
input_data = torch.randn(1, 1, 32, 32)
net_output = net(input_data)
print(f"Output: {net_output}")

# Loss function
target = torch.randn(10)
target = target.view(1, -1)  # Same shape as output
criterion = nn.MSELoss()

loss = criterion(net_output, target)
print(f"Loss function: {loss}")


# Backpropagation
# Zeroes the gradient buffers of all parameters.
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# Manual weighting update
learning_rate = 0.01
# for parameter in net.parameters():
#     f.data.sub_(parameters * learning_rate)


# Own optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

# Inside the training loop we would do this.
# optimizer.zero_grad()
# output = net(input_data)
# loss = criterion(net_output, target)
# loss.backward()
# optimizer.step()

# Loading and normalizing CIFAR10
transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
# Train
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# Test
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

