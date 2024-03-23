from django.db import models

# Create your models here.

class clothes_classification(models.Model):
    type = models.CharField(max_length=100)
    style = models.CharField(max_length=100)
    clothes_num = models.IntegerField()
