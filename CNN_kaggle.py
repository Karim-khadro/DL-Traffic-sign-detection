import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_feature, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature * 2, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(self.n_feature *2 , self.n_feature*4, kernel_size=5, padding=1)

        self.fc4 = nn.Linear(self.n_feature*4*2*2, output_size)
    
        self.Norm2dx1 = nn.BatchNorm2d(self.n_feature)
        self.Norm2dx2 = nn.BatchNorm2d(self.n_feature*2)
        self.Norm2dx4 = nn.BatchNorm2d(self.n_feature*4)

    def forward(self, f):
        out = self.conv1(f)
        out = F.relu(out)
        out = self.Norm2dx1(out)
        out = F.max_pool2d(out, 2)

        out = self.conv2(out)
        out = F.relu(out)
        out = self.Norm2dx2(out)
        out = F.max_pool2d(out, 2)

        out = self.conv3(out)
        out = F.relu(out)
        out = self.Norm2dx4(out)
        out = F.max_pool2d(out, 2)
   
        out = out.view(out.size(0), -1)
        
        out = self.fc4(out)
        out = F.softmax(out, dim = 1)
        return out