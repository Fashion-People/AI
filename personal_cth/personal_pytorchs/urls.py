from django.urls import path,include
from rest_framework import routers
from . import views #views.pu import

#인스턴스 이름 바꿀 수 있는 곳
router = routers.DefaultRouter()
router.register('clothes_classification',views.ItemViewSet)

urlpatterns = [
    path('',include(router.urls))
]

#기존에 있던 것 
#app_name ='personal_pytorchs'

#urlpatterns = [
#    path('',views.testapp_index,name='testapp_index')
#]

