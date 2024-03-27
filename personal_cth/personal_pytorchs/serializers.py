from rest_framework import serializers
from .models import clothes_classification

class ClothesSerializer(serializers.ModelSerializer):
    class Meta:
        model = clothes_classification
        fields = ("__all__")

class Clothes_url_Serializer(serializers.ModelSerializer):
    class Meta:
        model = clothes_classification
        fields = ('imageUrl','tempNumber')

class Clothes_analysis_Serializer(serializers.ModelSerializer):
    class Meta:
        model = clothes_classification
        fields = ('ClothesNumber','ClothesType','ClothesStyle')