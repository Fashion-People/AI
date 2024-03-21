import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import time
import matplotlib.pyplot as plt
import os

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device 객체
print(device)

#print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

fontpath = 'C:/Windows/Fonts/NanumGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=10).get_name()
plt.rc('font', family=font)


transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])

data_dir='C:/Users/kangb/capstoneDesign/Back-end/IdeaProjects/Image'

train_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), transforms_train)

train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=17, shuffle=True, num_workers=0)
#valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=4, shuffle=False, num_workers=0)

print('학습 데이터셋 크기:', len(train_datasets))
#print('테스트 데이터셋 크기:', len(valid_datasets))

class_names = train_datasets.classes
print('학습 클래스:', class_names)

def imshow(input, title):
    # torch.Tensor를 numpy 객체로 변환
    input = input.numpy().transpose((1, 2, 0))
    # 이미지 정규화 해제하기
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    # 이미지 출력
    plt.imshow(input)
    plt.title(title)
    plt.show()


model = models.resnet34(pretrained=True)
num_features = model.fc.in_features
# 전이 학습(transfer learning): 모델의 출력 뉴런 수를 3개로 교체하여 마지막 레이어 다시 학습
# transfer learning
model.fc = nn.Sequential(     
    nn.Linear(num_features, 256),        # 마지막 완전히 연결된 계층에 대한 입력은 선형 계층, 256개의 출력값을 가짐
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, num_features),      # Since 10 possible outputs = 10 classes
    nn.LogSoftmax(dim=1)              # For using NLLLoss()
)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)

model = model.to(device)

num_epochs = 30

best_epoch = None
best_loss = 5

''' Train '''
# 전체 반복(epoch) 수 만큼 반복하며
for epoch in range(num_epochs):
    model.train()
    start_time = time.time()
    
    running_loss = 0.
    running_corrects = 0

    # 배치 단위로 학습 데이터 불러오기
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 모델에 입력(forward)하고 결과 계산
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_datasets)
    epoch_acc = running_corrects / len(train_datasets) * 100.

    # 학습 과정 중에 결과 출력
    print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_epoch = epoch
        print("best_loss: {:.4f} \t best_epoch: {}".format(best_loss, best_epoch))


os.makedirs('./weight',exist_ok=True)
torch.save(model, './weight/model_best_epoch.pt')

