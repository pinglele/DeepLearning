import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)
])
train_Data = datasets.MNIST(
    root = 'D:/Jupyter/dataset/mnist/',
    train = True,
    download = True,
    transform = transform
)
test_Data = datasets.MNIST(
    root = 'D:/Jupyter/dataset/mnist/',
    train = False,
    download = True,
    transform = transform
)
train_loader = DataLoader(train_Data, shuffle=True,batch_size=128)
test_loader = DataLoader(test_Data, shuffle=True,batch_size=512)
class Inception(nn.Module):
    def __init__(self,in_channels):
        super(Inception,self).__init__()
        self.branch1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Conv2d(in_channels, 24, kernel_size=1)
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1,branch2,branch3,branch4]
        return torch.cat(outputs,1)#cat连接图片时图片尺寸要一致
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,10,kernel_size=5),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            Inception(in_channels=10),
            nn.Conv2d(88,20,kernel_size=5),nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            Inception(in_channels=20),
            nn.Flatten(),
            nn.Linear(1408,10)
        )
    def forward(self,x):
        y = self.net(x)
        return y
X = torch.rand(size = (1, 1, 28, 28))
for layer in CNN().net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
model = CNN().to('cuda:0')
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate
)
epochs = 10
losses = []
for epoch in range(epochs):
    for (x,y) in train_loader:
        x,y = x.to('cuda:0'), y.to('cuda:0')
        Pred=model(x)
        loss = loss_fn(Pred,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
Fig = plt.figure()
plt.plot(losses)
plt.show()
#test net
correct = 0
total = 0
with torch.no_grad():
    for (x,y) in test_loader:
        x,y = x.to('cuda:0'), y.to('cuda:0')
        Pred=model(x)
        _, predicted = torch.max(Pred.data,1)
        total += y.size(0)
        correct += torch.sum((predicted == y))

print(f'Accuracy of the network on the 10000 test images: {100*correct/total} %')