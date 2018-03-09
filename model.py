
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

#CNN network
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        #convolutional layers
        #the input is 3 RBM, the output is 64 because we want 64 filters 3x3
        self.conv1 = nn.Conv1d(3, 64, 3, padding = 1)
        #the input is the previous 64 filters and output is 64 filters, with 3x3
        self.conv2 = nn.Conv1d(64, 64, 3, padding = 1)
        self.conv3 = nn.Conv1d(64, 128, 3, padding = 1)
        self.conv4 = nn.Conv1d(128, 128, 3, padding = 1)
        self.conv5 = nn.Conv1d(128, 256, 3, padding = 1)
        self.conv6 = nn.Conv1d(256, 256, 3, padding = 1)

        #batch normalization
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(256)

        #max pooling
        self.mp = nn.MaxPool2d(2, stride=2) #2X2 with stride 2

        self.fc = nn.Linear(256*4*4, 500) #fully connected
        self.fc2 = nn.Linear(500, 10) #fully connected with classes 10

    def forward(self, inputs, hidden):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = self.mp(F.relu(self.batchnorm1(self.conv2(x))))

        x = F.relu(self.batchnorm2(self.conv3(x)))
        x = self.mp(F.relu(self.batchnorm2(self.conv4(x))))

        x = F.relu(self.batchnorm3(self.conv5(x)))
        x = self.mp(F.relu(self.batchnorm3(self.conv6(x))))

        x = x.view(x.size(0), -1) #flatten for fc
        #print (x.size(1))
        x = F.relu(self.fc(x))
        x = self.fc2(x)
        return F.log_softmax(x) #softmax classifier

net = RNN()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #stochastic gradient descent
#optimizer = optim.RMSprop(net.parameters(), lr = 0.0001)
optimizer = optim.Adam(net.parameters(), lr=0.001)

net.cuda()
