import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
'''import matplotlib.pyplot as plt
plt.ioff()  # 防止在非交互式环境意外打开图形窗口
plt.switch_backend('svg')  # 将后端设置为支持SVG输出的一个（如果有这样的后端的话）'''
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.Normalize(0.1307, 0.3081)
])
train_Data = datasets.MNIST(
    root = 'D:/Jupyter/dataset/mnist/',
    train = True,
    download = True,
    transform = transform
)
test_Data = datasets.MNIST(
    root = 'D:/Jupyter/dataset/mnist',
    train = False,
    download = True,
    transform = transform
)
train_loader = DataLoader(train_Data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_Data, batch_size=128, shuffle=False)
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,96,kernel_size = 11,stride = 4,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3,stride = 2),
            nn.Conv2d(96,256,kernel_size = 5,padding =2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =3,stride = 2),
            nn.Conv2d(256,384,kernel_size =3,padding = 1),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size = 3,padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =3,stride = 2),
            nn.Flatten(),
            nn.Linear(6400,4096),nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )
    def forward(self, x):
        y = self.net(x)
        return y

X = torch.randn(size = (1, 1, 224, 224))
for layer in CNN().net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:',X.shape)
model = CNN().to('cuda:0')
loss_fn = nn.CrossEntropyLoss()

learning_rate = 0.01
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = learning_rate,
)
epochs = 10
losses = []
for epoch in range(epochs):
    for (x,y) in train_loader:
        x,y = x.to('cuda:0'),y.to('cuda:0')
        Pred = model(x)
        loss = loss_fn(Pred,y)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
Fig = plt.figure()
plt.plot(range(len(losses)),losses)
plt.show()
#测试网络
correct = 0
total = 0
with torch.no_grad():
    for (x,y) in test_loader:
        x,y = x.to('cuda:0'),y.to('cuda:0')
        Pred = model(x)
        _,prediced = torch.max(Pred.data,dim = 1)
        correct += torch.sum((prediced == y))
        total += y.size(0)
print(f"Accuracy of the network on the 10000 test images:{100*correct/total}%")