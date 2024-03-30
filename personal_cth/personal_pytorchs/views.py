# Create your views here.
from django.core.files.storage import FileSystemStorage
import requests
import json

from django.shortcuts import render
# DRF 관련 import 
from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view

#api 시리얼 import
from .serializers import ClothesSerializer
from .serializers import Clothes_url_Serializer
from .serializers import Clothes_analysis_Serializer
#api 모델 임포트 import
from .models import clothes_classification 

#spring boot 연동을 위한 import
from django.http import JsonResponse


#class ClothesViewSet(viewsets.ModelViewSet):
    #queryset = clothes_classification.objects.all()
    #serializer_class = ClothesSerializer

#미리 해보는 것 연습용
@api_view(['GET'])
def API(request):
    return Response("api 에 오신것을 환영합니다.")


# 이미지 url 가져오기 & json 형태 데이터 파싱 

class ImageAnalysis(APIView):
     def get(self, request):
        #파라미터 가져오기
        url = request.query_params.get('imageUrl', None)
        tempNumber = request.query_params.get('tempNumber',None)

        url = 'https://media.bunjang.co.kr/product/242654539_1_1699790685_w360.jpg'

        if url :
            ClothesNumber, ClothesType ,ClothesStyle =  predictImage(url)
            return Response({'ClothesNumber': ClothesNumber,'ClothesType':ClothesType,'ClothesStyle':ClothesStyle})
        else :
            return Response({'error': 'URL parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)





#챗지피티 참조
#data_to_send = {'key': 'value'}
#response = requests.post('http://localhost:8080/data', json=data_to_send)
#return JsonResponse({'status': response.status_code})





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



#기존 url + predictImage를 더해서 request 했을때를 의미함
def predictImage(imageUrl):
    #print(request)
    #print (request.POST.dict())
    #print (request.FILES['filePath'])
    #file obj => 파일 이름

    #fileobj = (request.FILES['filePath'])
    #fs = FileSystemStorage()
    #filePathName = fs.save(fileobj.name,fileobj)
    #filePathName = fs.url(filePathName)
    
    #반팔
    #image_url='https://image.msscdn.net/images/goods_img/20230329/3188053/3188053_16813635662783_500.jpg' 
    
    #트렌치 코트 
    #image_url='https://media.bunjang.co.kr/product/242654539_1_1699790685_w360.jpg'


    #블라우스
    #image_url='https://media.bunjang.co.kr/product/124189638_1_1589021004_w360.jpg'

    #image_url='https://media.bunjang.co.kr/product/224787189_1_1684729265_w360.jpg'

    #image_url='https://media.bunjang.co.kr/product/246083941_1_1702366323_w360.jpg'

    #야상
    #image_url='https://qi-o.qoo10cdn.com/goods_image_big/3/6/9/3/7877613693_l.jpg'

    #후드티
    #image_url='https://images.kolonmall.com/Prod_Img/CJ/2021/LM6/J3TEA21703GYM_LM6.jpg'

    #image_url='https://m.editail.com/web/product/big/202301/111c3a3ccc4a496fcea47fb1a0fad190.jpg'
    #니트
    #image_url='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcThL2z9FRqnYfqVhp9MfahfmYod4WMSlp8qWA&usqp=CAU'
    #image_url='https://web.joongna.com/cafe-article-data/live/2024/01/15/1035339901/1705305954360_000_DXO5P_main.jpg'
    
    #장고 서버 열면, 여기 url 수정 필요 
    #response = requests.get('http://127.0.0.1:8000/clothes_urlAPI/')
    #data_from_springboot = response.json()
    #imageUrl = data_from_springboot.get('imageUrl')
    #tempNumber = data_from_springboot.get('tempNumber')

    
    response = requests.get(imageUrl)
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


    #이부분에 post 하는 def 시켜놓은거 불러다가 저장해주기?

    #context={'filePathName':filePathName,'predictImage':predictImage}
    #기존 코드 밑에 2줄
    #context={'filePathName':imageUrl,'predictImage':predictImage,'styleImage':Style_Image}
    #return render(request,'index.html',context)
        
    tempNumber =1


    return tempNumber,predictImage,Style_Image


