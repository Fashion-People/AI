from django.urls import path
from . import views

app_name ='personal_pytorchs'

urlpatterns = [
    path('',views.testapp_index,name='testapp_index')
]

