import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTNet51(nn.Module):
    """
        CNN for LSTCam images
        5 CL (the last is a kind of FC : 1 image feature to 1 pixel), 4 + maxPooling and batchNorm
        1 FC
    """
    def __init__(self, output_size):
        super(LSTNet51, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu1 = nn.ReLU()
        # maxpooling 1, by default floor
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # nn.AvgPool2d
        # batch norm
        self.batchnorm1 = nn.BatchNorm2d(16)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu2 = nn.ReLU()
        # maxpooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # batch norm
        self.batchnorm2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu3 = nn.ReLU()
        # maxpooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # nn.AvgPool2d
        # batch norm
        self.batchnorm3 = nn.BatchNorm2d(64)

        # conv4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu4 = nn.ReLU()
        # maxpooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # nn.AvgPool2d
        # batch norm
        self.batchnorm4 = nn.BatchNorm2d(128)

        # conv5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        # non-linearity
        self.relu5 = nn.ReLU()

        # readout, regression of energy, altitude, azimuth, xCore, yCore
        # self.fc1 = nn.Linear(128, 5)
        # readout, regression of xCore and yCore
        self.fc1 = nn.Linear(128, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data, mode='fan_out')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.batchnorm2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = self.batchnorm3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        out = self.batchnorm4(out)
        out = self.conv5(out)
        out = self.relu5(out)

        # Reshape out from 100,128,1 to 100,128
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out


class LSTNet512(nn.Module):
    """
        CNN for LSTCam images
        5 CL (the last is a kind of FC : 1 image feature to 1 pixel), 2 + maxPooling and 4 + batchNorm
        1 FC
    """
    def __init__(self, output_size):
        super(LSTNet512, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu1 = nn.ReLU()
        # batch norm
        self.batchnorm1 = nn.BatchNorm2d(16)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu2 = nn.ReLU()
        # maxpooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # batch norm
        self.batchnorm2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu3 = nn.ReLU()
        # batch norm
        self.batchnorm3 = nn.BatchNorm2d(64)

        # conv4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu4 = nn.ReLU()
        # maxpooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # nn.AvgPool2d
        # batch norm
        self.batchnorm4 = nn.BatchNorm2d(128)

        # conv5
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu5 = nn.ReLU()

        # readout
        self.fc1 = nn.Linear(13*13*128, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data, mode='fan_out')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.batchnorm2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batchnorm3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)
        out = self.batchnorm4(out)
        out = self.conv5(out)
        out = self.relu5(out)

        # Reshape out from 100,128,1 to 100,128
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        return out


class LSTNet42(nn.Module):
    """
        CNN for LSTCam images
        4 CL , 2 + maxPooling and 3 + batchNorm
        2 FC
    """
    def __init__(self, output_size):
        super(LSTNet42, self).__init__()
        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu1 = nn.ReLU()
        # batch norm
        self.batchnorm1 = nn.BatchNorm2d(16)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu2 = nn.ReLU()
        # maxpooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        # batch norm
        self.batchnorm2 = nn.BatchNorm2d(32)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu3 = nn.ReLU()
        # batch norm
        self.batchnorm3 = nn.BatchNorm2d(64)

        # conv4
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # non-linearity
        self.relu4 = nn.ReLU()
        # maxpooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)  # nn.AvgPool2d

        # readout,
        self.fc1 = nn.Linear(13*13*128, 256)
        self.fc2 = nn.Linear(256, output_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data, mode='fan_out')

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.batchnorm1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.batchnorm2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = self.batchnorm3(out)
        out = self.conv4(out)
        out = self.relu4(out)
        out = self.maxpool4(out)

        # Reshape out from 100,128,1 to 100,128
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        return out