# In[1]:



import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os


class RoadSignDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, df, transform):
        """Initializes a dataset containing images and labels."""
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.df = df
        
    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.df)
        
    def __getitem__(self, index):
        """Returns the index-th data item of the dataset."""
        x = True
        i = 1
        while x:
            try:
                img = torchvision.datasets.folder.pil_loader(os.path.join(self.root_dir, self.df["path"][index+i%len(self.df)]))
                x =False
            except:
                i += 1
                
        if self.transform is not None:
            img = self.transform(img)
        label = self.df["class"][index]
        
        return img,label


headers = ["path", "class"]
df = pd.read_csv("csv/Train.csv")
df1 = pd.read_csv("csv/Test.csv")

data = [df["path"],df["class"]]
trainDf = pd.concat(data, axis=1, keys=headers)

data = [df1["path"],df1["class"]]
testDF = pd.concat(data, axis=1, keys=headers)     


# In[4]:


# Get mean & std of training images

headers = ["path", "class"]
t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
mean_train = 0.0
std_train = 0.0

testset = RoadSignDataset('Train', trainDf,transform=t)  # test directory
loader_test = torch.utils.data.DataLoader(testset, batch_size=len(trainDf["path"]), num_workers=0)#len(trainDf)
data_test = next(iter(loader_test))
mean_train = data_test[0].mean()
std_train = data_test[0].std()

print(mean_train)
print(std_train)

# tensor(0.3373)
# tensor(0.2890)


# In[5]:


# Get mean & std of testing images
t = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor() ])
testset = RoadSignDataset('Test', testDF,transform=t)  # test directory
loader_test = torch.utils.data.DataLoader(testset, batch_size=len(testDF), num_workers=0)#len(trainDf)
data_test = next(iter(loader_test))

mean_test = data_test[0].mean()
std_test = data_test[0].std()
data_test[0].mean(), data_test[0].std()

print(mean_test)
print(std_test)

#tensor(0.3653)
#tensor(0.2976)


n_feature = 27
output_dim = 80
learning_rate =1.0
num_epochs = 10

class CNN(nn.Module):
    def __init__(self, n_feature, output_size):
        super(CNN, self).__init__()
        self.n_feature = n_feature
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.n_feature, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(self.n_feature, self.n_feature * 2, kernel_size=5, padding=1)
        self.conv3 = nn.Conv2d(self.n_feature *2 , self.n_feature*4, kernel_size=5, padding=1)

        self.fc1 = nn.Linear(self.n_feature*4*2*2, output_size)
    
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
        
        out = self.fc1(out)
        out = F.softmax(out, dim = 1)
        
        return out

device = "cuda:0"
# Reading inputs
mean_test = torch.tensor(0.3653)
std_test = torch.tensor(0.2976)
transform_test = transforms.Compose([transforms.Resize((32,32)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean_test,std_test)])
mean_train = torch.tensor(0.3373)
std_train = torch.tensor(0.2890)
transform_train = transforms.Compose([transforms.Resize((32,32)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean_train,std_train)])

trainset = RoadSignDataset('Train', trainDf, transform=transform_train)  # training directory
testset = RoadSignDataset('Test', testDF,transform=transform_test)  # test directory

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)


# Model
model = CNN(n_feature, output_dim)
print(model)
model.to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, rho=0.9, eps=1e-06, weight_decay=0)
criterion = nn.NLLLoss()  


epochs_train_loss = []
epochs_test_loss = []
accuracy_train = []
accuracy_test = []

writer = SummaryWriter(f"runs/MNIST/Classifier")
stepTrain = 0
stepTest = 0

for i in range(num_epochs):
    accuracy = 0
    tmp_loss = []
    print(i+1)
    for inputs, labels in trainloader:
        model.train()
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = model(inputs)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        corr = ((output.argmax(dim=1) == labels).float().mean())
        accuracy += corr/len(trainloader)
        tmp_loss.append(loss.detach())

        writer.add_scalar("Training loss", loss.item(), global_step=stepTrain)
        writer.add_scalar("Training accuracy", accuracy *100, global_step=stepTrain)
        stepTrain += 1
        
        train_loss = (sum(tmp_loss)/len(tmp_loss))

    print("TRAIN : epoch : {}  accuracy : {}%  loss : {}".format(i+1, (accuracy*100), (sum(tmp_loss)/len(tmp_loss))))
    
    epochs_train_loss.append((sum(tmp_loss)/len(tmp_loss)).cpu())
    accuracy_train.append((accuracy*100).cpu())

    with torch.no_grad():
        correct = 0
        total = 0
        tmp_loss = []
        for inputs, labels in testloader:
            model.eval()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            tmp_loss.append(loss.detach())
            
#           Tensorbroard 
            writer.add_scalar("Testing loss", loss, global_step=stepTest)
            writer.add_scalar("Testing accuracy", (100 * correct / total), global_step=stepTest)
            stepTest += 1

        print("TEST : epoch : {}  accuracy : {}%  loss : {}".format(i+1,  (100 * correct / total), (sum(tmp_loss)/len(tmp_loss))))
        epochs_test_loss.append((sum(tmp_loss)/len(tmp_loss)).cpu())
        accuracy_test.append((100 * correct / total))
        
#   Save the model after every epoch
    checkpoint = {
        'epoch': i + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(checkpoint, "backup/classifier_epoch" + str(i + 1))

plt.plot(epochs_train_loss, label="Train")
plt.plot(epochs_test_loss, label="Test")
plt.title("Loss (training & testing)")
plt.ylabel("Mean loss")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.show()

plt.plot(accuracy_train, label="Train")
plt.plot(accuracy_test, label="Test")
plt.title("Accueacy (training & testing)")
plt.ylabel("Accueacy %")
plt.xlabel("No. epoch")
plt.legend(loc="upper left")
plt.show()

from torchinfo import summary
batch_size = 128
summary(model, input_size=(batch_size, 3, 32, 32))
