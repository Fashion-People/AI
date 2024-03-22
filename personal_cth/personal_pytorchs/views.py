from django.shortcuts import render
# Create your views here.

from django.core.files.storage import FileSystemStorage

import requests

#model을 위한 준비과정
import io
import time
import json
import torch 
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO

from torchvision import models
from torchvision import transforms

from django.conf import settings
#from PIL import image

import torch.nn as nn
import torch.optim as optim

from PIL import Image

device = torch.device('cpu')

img_height, img_width = 224,224

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

with open('./models/imagenet_classes.json','r') as f:
    labelInfo=f.read()
labelInfo=json.loads(labelInfo)

model_path='./models/model_best_epoch3.pth'
loaded_model = torch.load(model_path)

model_path='./models/style_only_model_best_epoch3.pth'
style_loaded_model = torch.load(model_path)

model = loaded_model
style_model = style_loaded_model
#num_features = model.fc.in_features
# 전이 학습(transfer learning): 모델의 출력 뉴런 수를 3개로 교체하여 마지막 레이어 다시 학습
#model.fc = nn.Linear(num_features, 17)
#model = model.to(device)

#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#기존에 있던 것
#def index(request):
#    context={'a':1}
#    return render(request,'index.html',context)

#이미지 분석전용?
def index(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print(request)
    print (request.POST.dict())
    #print (request.FILES['filePath'])
    #file obj => 파일 이름

    
    #fileobj = (request.FILES['filePath'])
    #fs = FileSystemStorage()
    #filePathName = fs.save(fileobj.name,fileobj)
    #filePathName = fs.url(filePathName)
    
    #image_url = 'https://fashionbucket.s3.ap-northeast-2.amazonaws.com/profile/image/20240128190555194.jpg'
    #image_url = 'http://127.0.0.1:8000/media/slxm_OLbTRZH.jpg'
    #image_url = 'https://media.bunjang.co.kr/product/224788073_1_1684729916_w360.jpg'
    
    #반팔
    #image_url='https://image.msscdn.net/images/goods_img/20230329/3188053/3188053_16813635662783_500.jpg' 
    
    #트렌치 코트 
    #image_url='https://media.bunjang.co.kr/product/242654539_1_1699790685_w360.jpg'


    #블라우스
    #image_url='https://media.bunjang.co.kr/product/124189638_1_1589021004_w360.jpg'

    image_url='https://media.bunjang.co.kr/product/224787189_1_1684729265_w360.jpg'

    #image_url='https://media.bunjang.co.kr/product/246083941_1_1702366323_w360.jpg'

    #야상
    #image_url='https://qi-o.qoo10cdn.com/goods_image_big/3/6/9/3/7877613693_l.jpg'

    #후드티
    #image_url='https://images.kolonmall.com/Prod_Img/CJ/2021/LM6/J3TEA21703GYM_LM6.jpg'

    #image_url='https://m.editail.com/web/product/big/202301/111c3a3ccc4a496fcea47fb1a0fad190.jpg'
    #니트
    #image_url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThL2z9FRqnYfqVhp9MfahfmYod4WMSlp8qWA&usqp=CAU'
    #image_url='https://web.joongna.com/cafe-article-data/live/2024/01/15/1035339901/1705305954360_000_DXO5P_main.jpg'
    response = requests.get(image_url)
    image_bytes = response.content

    # BytesIO를 사용하여 이미지 불러오기
    image = Image.open(io.BytesIO(image_bytes))
    #image_path = "./media/img.jpg"
    #image.save(image_path)

    image = transforms_test(image).unsqueeze(0).to(device)
    # 이미지 표시
    #image.show()

    #fileobj = (request.FILES['filePath'])
    #fs = FileSystemStorage()
    #filePathName = fs.save(saved_image_path)
    #filePathName = fs.url(filePathName)
    
    #image_bytes = fileobj.read()

    #image = Image.open(io.BytesIO(image_bytes))
    #image = transforms_test(image).unsqueeze(0).to(device)

    #up_image = Image.open(io.BytesIO(image_bytes))
    #up_image.save("./static/img.jpg","jpeg")
    
    #print(filePathName)
    #testimage='.'+image_bytes
    #img = Image.open(testimage)
    #x=imshow(testimage)


    with torch.no_grad():
       model.eval()
       output=model(image)
       style_model.eval()
       style_output=style_model(image)
    # 이미지 저장하는 것 (현재 파일에 올려둔 이미지)
    #context={'filePathName':filePathName}
       
    #predictImage =torch.argmax(output[0]).item()
    _, preds = torch.max(output, 1)

    if (preds[0].item()==0):
        predictImage='가디건'
    elif (preds[0].item()==1):
        predictImage='긴팔 티'
    elif (preds[0].item()==2):
        predictImage='누빔 옷'
    elif (preds[0].item()==3):
        predictImage='니트'
    elif (preds[0].item()==4):
        predictImage='린넨 옷'
    elif (preds[0].item()==5):
        predictImage='맨투맨'
    elif (preds[0].item()==6):
        predictImage='민소매'
    elif (preds[0].item()==7):
        predictImage='반팔'
    elif (preds[0].item()==8):
        predictImage='블라우스'
    elif (preds[0].item()==9):
        predictImage='야상'
    elif (preds[0].item()==10):
        predictImage='얇은 셔츠'
    elif (preds[0].item()==11):
        predictImage='자켓'
    elif (preds[0].item()==12):
        predictImage='청자켓'
    elif (preds[0].item()==13):
        predictImage='코트'
    elif (preds[0].item()==14):
        predictImage='트렌치코트'
    elif (preds[0].item()==15):
        predictImage='패딩'
    elif (preds[0].item()==16):
        predictImage='후드티'



    _, preds = torch.max(style_output, 1)

    if (preds[0].item()==0):
        Style_Image='모던'
    elif (preds[0].item()==1):
        Style_Image='스포티'
    elif (preds[0].item()==2):
        Style_Image='캐주얼'
    elif (preds[0].item()==3):
        Style_Image='페미닌'


    #context={'filePathName':filePathName,'predictImage':predictImage}
    context={'filePathName':image_url,'predictImage':predictImage,'styleImage':Style_Image}
    return render(request,'index.html',context)

