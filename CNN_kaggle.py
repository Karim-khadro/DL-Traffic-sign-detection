import torch, torchvision
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_feature, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature, kernel_size=5, padding=1)
        
        self.conv3 = nn.Conv2d(self.n_feature, self.n_feature*2, kernel_size=5, padding=1)
        self.conv4 = nn.Conv2d(self.n_feature*2, self.n_feature*2, kernel_size=5, padding=1)
        
        self.conv5 = nn.Conv2d(self.n_feature*2, self.n_feature*4, kernel_size=5, padding=1)
        self.conv6 = nn.Conv2d(self.n_feature*4, self.n_feature*4, kernel_size=5, padding=1)

#         self.conv7 = nn.Conv2d(self.n_feature*4, self.n_feature*4, kernel_size=3, padding=1)
#         self.conv8 = nn.Conv2d(self.n_feature*4, self.n_feature*2, kernel_size=3, padding=1)
        
#         self.fc1 = nn.Linear(self.n_feature*4*4*4, 400)
#         self.fc2 = nn.Linear(self.n_feature*4*4*2, self.n_feature*4*4)
#         self.fc3 = nn.Linear(self.n_feature*4*4, 128)
        self.fc4 = nn.Linear(self.n_feature*4*4*4, output_size)

        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.5)
        self.dropout2d = nn.Dropout2d(p=0.001)
        self.LeakyReLU = nn.LeakyReLU()
        self.Relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.tanhActivation = nn.Tanh()
        
        self.Norm2dx1 = nn.BatchNorm2d(self.n_feature)
        self.Norm2dx2 = nn.BatchNorm2d(self.n_feature*2)
        self.Norm2dx4 = nn.BatchNorm2d(self.n_feature*4)

    def forward(self, f):
#         out = self.conv1(f)
#         out = nn.functional.relu(self.conv1(f))
#         out = self.pool (out)
        out = self.Norm2dx1(self.Relu(self.conv1(f)))

#         out = self.conv2(out)
#         out = nn.functional.relu(self.conv2(out))
#         out = self.Norm2dx1(nn.functional.relu(self.conv2(out)))
#         out = self.pool(self.Norm2dx1(nn.functional.relu(self.conv2(out))))
        out = self.pool(self.Norm2dx1(self.Relu(self.conv2(out))))

#         out = self.conv3(out)
#         out = nn.functional.relu(self.conv3(out))

        out = self.Norm2dx2(self.Relu(self.conv3(out)))
    
#         out = self.pool(out)

#         out = self.conv4(out)
#         out = nn.functional.relu(self.conv4(out))
#         out = self.Norm2dx2(nn.functional.relu(self.conv4(out)))
#         out = self.pool(self.Norm2dx2(nn.functional.relu(self.conv4(out))))
        out = self.pool(self.Norm2dx2(self.Relu(self.conv4(out))))

#         out = self.conv5(out)
#         out = nn.functional.relu(self.conv5(out))

        out = self.Norm2dx4(self.Relu(self.conv5(out)))
    
#         out = self.pool(out)

#         out = self.conv6(out)
#         out = nn.functional.relu(self.conv6(out))
#         out = self.Norm2dx4(nn.functional.relu(self.conv6(out)))

        out = self.pool(self.Norm2dx4(self.Relu(self.conv6(out))))
    
    
#         out = self.Norm2dx4(self.ReLU(self.conv7(out))) 
        

        
#         out = self.dropout2d(self.pool(self.Norm2dx2(self.ReLU(self.conv8(out)))))
        
        
        out = out.view(out.size(0), -1)
        
#         out = self.fc1(out)
#         out = nn.functional.relu(self.fc1(out))

#         out = self.dropout(self.ReLU(self.fc1(out)))
        
#         out = self.fc2(out)
#         out = nn.functional.relu(self.fc2(out))

#         out = self.dropout(nn.functional.relu(self.fc2(out)))
        
#         out = self.fc3(out)
#         out = nn.functional.relu(self.fc3(out))

#         out = self.dropout(nn.functional.relu(self.fc3(out)))
        
#         out = self.fc4(out)
        out = self.softmax(self.fc4(out))
        
        return out