"""personal_cth URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from personal_pytorchs import views
from django.urls import re_path as url
from django.conf.urls.static import static
from django.conf import settings 

from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework.permissions import AllowAny


schema_view=get_schema_view(
    openapi.Info(
        title="personal_clothes API",
        default_version="v1",
        description="옷 이미지 분류 값",
        terms_of_service="https://cholol.tistory.com/551",
    ),
    public=True,
    permission_classes=(AllowAny,),
)

urlpatterns = [
    #장고 부분
    path('admin/', admin.site.urls),
    #url('^$',views.index,name='homepage'),
    url('predictImage',views.predictImage,name='predictImage'),

    #스웨거 부분
    url(r'swagger(?P<format>\.json|\.yaml)', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    url(r'swagger', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    url(r'redoc', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),

    #api 부분
    path("",include("personal_pytorchs.urls")), #personal_pytorchs/urls.py 사용한다

    
]

#media 파일로 들어간다.
#urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
