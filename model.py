import torch.nn as nn
import torch

class Signet(nn.Module):
    def __init__(self, num_users=10):
        super(Signet, self).__init__()
        self.num_classes = num_users

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.norm1 = nn.BatchNorm2d(96)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.norm2 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(384)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.norm4 = nn.BatchNorm2d(384)
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.norm5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.act5 = nn.ReLU()

        self.fc6 = nn.Linear(in_features=3*5*256, out_features=2048)
        self.norm6 = nn.BatchNorm1d(2048)
        self.act6 = nn.ReLU()

        self.fc7 = nn.Linear(in_features=2048, out_features=2048)
        self.norm7 = nn.BatchNorm1d(2048)
        self.act7 = nn.ReLU()

        # probability that image belongs to each user (sum=1)
        self.fc_m = nn.Linear(in_features=2048, out_features=self.num_classes)
        self.act_m = nn.Softmax(1)
        # probability that image is a forgery
        # self.fc_f = nn.Linear(in_features=2048, out_features=1)
        # self.act_f = nn.Sigmoid()

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        x = x.unsqueeze_(1)

        x = self.act1(self.pool1(self.norm1(self.conv1(x))))

        x = self.act2(self.pool2(self.norm2(self.conv2(x))))

        x = self.act3(self.norm3(self.conv3(x)))

        x = self.act4(self.norm4(self.conv4(x)))

        x = self.act5(self.pool5(self.norm5(self.conv5(x))))

        x = x.view(-1, 256*3*5)

        x = self.act6(self.norm6(self.fc6(x)))
        x = self.act7(self.norm7(self.fc7(x)))

        m = self.act_m(self.fc_m(x))
        # f = self.act_f(self.fc_f(x))

        # tuple of tensors
        # return(m,f)
        return m
