from django.db import models

# Create your models here.

class Service(models.Model):
    name = models.CharField(max_length=100)
    host = models.CharField(max_length=100)
    port = models.IntegerField()
    metric_name = models.CharField(max_length=1000)
    epoch = models.IntegerField()

class TimeInterval(models.Model):
    time = models.IntegerField()
