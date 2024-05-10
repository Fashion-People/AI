# Create your views here.
from django.core.files.storage import FileSystemStorage
import requests
import json
from urllib.parse import quote

from django.shortcuts import render
# DRF 관련 import 
from rest_framework.views import APIView
from rest_framework import viewsets
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view



#spring boot 연동을 위한 import
from django.http import JsonResponse


# 이미지 url 가져오기 & json 형태 데이터 파싱 

class ImageAnalysis(APIView):
     
     def post(self, request, *args, **kwargs):
        #파라미터 가져오기
        #배열의 형태를 가져온다.
        try:
            data = json.loads(request.body.decode('utf-8'))
            tempNumber = data.get('tempNumber',None)
            data = data['imageUrl']
            print(data)

            result_data = [] #결과를 담을 배열
            for idx, item in enumerate(data, start=1):

                url = item
                ClothesType ,ClothesStyle =  predictImage(item)
                if ClothesType=='error':
                    return JsonResponse({"messaage":'Unidentified image error',"number":idx})
                else :
                    result_data.append({
                        'tempNumber' : tempNumber, #이미지 url 받았을 때 이미지 리스트 번호
                        'clothesNumber' : idx, #옷번호
                        'clothesStyle' : ClothesStyle, #옷 종류
                        'clothesType' : ClothesType,#옷 형태
                        'imageUrl' : url #옷 url 
                    })

            if url :
                return JsonResponse(result_data, safe=False)
            else :
                return Response({'error': 'URL parameter is missing'}, status=status.HTTP_400_BAD_REQUEST)
        except KeyError:
            return JsonResponse({"message": 'key error'}, status=400)
    
     #def get (self,request):
     #    return Response('전달 완료')
    



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
from PIL import UnidentifiedImageError
import urllib.request

from django.http import HttpRequest


device = torch.device('cpu')

img_height, img_width = 224,224

transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model_path='./models/model_best_epoch4.pth'
loaded_model = torch.load(model_path)

model_path='./models/style_only_model_best_epoch4.pth'
style_loaded_model = torch.load(model_path)

model = loaded_model 
style_model = style_loaded_model


#기존 url + predictImage를 더해서 request 했을때를 의미함
def predictImage(imageUrl):
   
    encodedImageUrl = quote(imageUrl, safe=':/')
   
    #이미지 다운로드 
    file_path = './personal_pytorchs/temp/practice.jpg'
    urllib.request.urlretrieve(encodedImageUrl,file_path)


    path = './personal_pytorchs/temp/practice.jpg'
    try:        
        image = Image.open(path)
        image = transforms_test(image).unsqueeze(0).to(device)
        with torch.no_grad():
            model.eval()
            output=model(image) #옷 종류 분석
            style_model.eval()
            style_output=style_model(image) #옷 스타일 분석
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

        return predictImage,Style_Image


    except UnidentifiedImageError as e:
        print("이미지 파일을 식별할 수 없습니다:", e)
        predictImage='error'
        Style_Image='error'
        return predictImage,Style_Image
       
    #image = Image.open(file_path)
    #image = Image.open(io.BytesIO(image_bytes))

    #image = transforms_test(image).unsqueeze(0).to(device)

    
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


   



    #이부분에 post 하는 def 시켜놓은거 불러다가 저장해주기?

    #context={'filePathName':filePathName,'predictImage':predictImage}
    #기존 코드 밑에 2줄
    #context={'filePathName':imageUrl,'predictImage':predictImage,'styleImage':Style_Image}
    #return render(request,'index.html',context)