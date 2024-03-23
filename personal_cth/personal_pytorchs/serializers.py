from rest_framework import serializers
from .models import clothes_classification

class ItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = clothes_classification
        fields = ("__all__")

