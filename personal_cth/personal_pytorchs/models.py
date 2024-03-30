from django.db import models

# Create your models here.

class clothes_classification(models.Model):
    #post에 이용
    ClothesNumber = models.IntegerField() #여러개 옷일때 구분
    ClothesType = models.CharField(max_length=100)
    ClothesStyle = models.CharField(max_length=100)
    #get에 이용
    imageUrl = models.TextField()
    tempNumber = models.IntegerField()

    
#모델이 추가됨