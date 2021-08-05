from rest_framework import serializers
from .models import  Service,TimeInterval

class ServiceSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    host = serializers.CharField(max_length=100)
    port = serializers.IntegerField()
    metric_name = serializers.CharField(max_length=1000)

    def create(self, validated_data):
        return Service.objects.create(**validated_data)

class TimeIntervalSerializer(serializers.Serializer):
    time = serializers.IntegerField()

    def create(self,validated_data):
        return TimeInterval.objects.create(**validated_data)

    def  update(self,instance,validated_date):
        instance.time = validated_date.get('time',instance.time)
        instance.save()
        return instance