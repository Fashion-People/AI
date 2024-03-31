from django.urls import path,include
from rest_framework import routers
from . import views #views.pu import
from .views import ImageAnalysis

#인스턴스 이름 바꿀 수 있는 곳
#라우터 기본(?) 127.0.0.1 했을때 나오도록 한 것 
router = routers.DefaultRouter()
#router.register('clothes_classification',views.ClothesViewSet)

urlpatterns = [
    path('',include(router.urls)),
    #실제 사용하는 api 
    path("imageAnalysis/",ImageAnalysis.as_view(),name='image_analysis'),
    
]

#기존에 있던 것 
#app_name ='personal_pytorchs'

#urlpatterns = [
#    path('',views.testapp_index,name='testapp_index')
#]

