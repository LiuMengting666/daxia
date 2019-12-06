# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 22:43:28 2019

@author: Lenovo
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from PIL import Image

transform = T.Compose([
    T.ToTensor(),  
])

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=310):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.fc = nn.Linear(4608, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18():
    return ResNet(ResidualBlock)

class Coding(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform   
        
    def one_hot(self, label):
        bp = torch.Tensor([])
        for i in range(len(label)):
            num = ord(label[i])-48
            if num>9:
                num -= 7
                if num>35:
                    num -= 6         
            a = torch.zeros(1, 62)
            a[:,num] = 1
            bp = torch.cat((bp,a),dim=1)
        return bp
        
    def __getitem__(self, index):
        image_path = self.paths[index]    
        label = list(image_path)[:-4]
        label = self.one_hot(label).reshape(310)

        pil_image = Image.open(self.root+image_path)
        if self.transforms:
            data = self.transforms(pil_image)
        else:
            image_array = np.asarray(pil_image)
            data = torch.from_numpy(image_array)
        return data, label

    def __len__(self):
        return len(self.paths)
class UnCoding(Dataset):
    
    def __init__(self, root, transform=None):
        self.root = root
        self.paths = os.listdir(root)
        self.transforms = transform   
        
    def __getitem__(self, index):
        image_path = self.paths[index]
        label = image_path
        if label !='.DS_Store':
            label = list(label)[:-4]
            label = int(''.join(label))
            pil_image = Image.open(self.root+image_path)   
            if self.transforms:
                data = self.transforms(pil_image)
            else:
                image_array = np.asarray(pil_image)
                data = torch.from_numpy(image_array)
            return data, label

    def __len__(self):
        return len(self.paths)

def uncode(code):
    result = list()
    for i in range(len(code)):
        if code[i]<10:
            result.append(chr(code[i]+48))
        elif 10<=code[i]<36:
            result.append(chr(code[i]+55))
        else: 
            result.append(chr(code[i]+61))
    return result


data = Coding('train/train/', transform)
dataloader = DataLoader(data, batch_size=32, shuffle=True, drop_last=False)
img, label = data[0]

cnn = ResNet18()

loss_fn = nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(cnn.parameters())


for i in range(6):  
    for j,(img,labels) in enumerate(dataloader):
        out = cnn(img)
        loss = loss_fn(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if j % 1000 == 0:
            print('i=%d j=%d Loss: %.5f' %(i,j,loss.item()))


torch.save(cnn.state_dict(),'parameter.pt')
data = UnCoding('test/test/', transform)
cnn = ResNet18()
cnn.load_state_dict(torch.load('parameter2.pt'))

cnn.eval()

result = dict()

for i in range(len(data)):
    imgs, labels = data[i]
    imgs = torch.Tensor(imgs).reshape(1,3,30,150)
    single_result = cnn(imgs)
    single_result = single_result.view(-1, 62)
    single_result = nn.functional.softmax(single_result, dim=1)
    single_result = torch.argmax(single_result, dim=1)
    out = uncode(single_result)
    result[labels] = out

    
index = list()
labels = list()
for i in range(len(result)):
    index.append(i)
    labels.append(''.join(result[i]))
np.savetxt('sample.csv',labels, fmt='%s')