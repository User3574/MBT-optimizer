import torch.nn as nn
import torch.nn.functional as ff


class SmallNet(nn.Module):
    def __init__(self, num_classes):
        self.num_classes = num_classes

        super(SmallNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, self.num_classes)

    def forward(self, x):
        x = ff.relu(ff.max_pool2d(self.conv1(x), 2))
        x = ff.relu(ff.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = ff.relu(self.fc1(x))
        x = ff.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def smallnet(num_classes=10):
    return SmallNet(num_classes)