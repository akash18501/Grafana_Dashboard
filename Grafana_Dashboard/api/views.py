from django.shortcuts import render
from django.http import HttpResponse
from .models import Service,TimeInterval
from rest_framework.parsers import JSONParser
from .serializers import  ServiceSerializer,TimeIntervalSerializer
from rest_framework.renderers import  JSONRenderer
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import io
import threading
from .CNN_model import metric_prediction

timeobject = TimeInterval.objects.get(id = 1).time
# Create your views here.

def home(request):
    return HttpResponse('<h1>Hello world</h1>')

@csrf_exempt
def AddService(request):
    if request.method == 'POST':
        json_data = request.body
        print("json data is" , json_data)
        stream = io.BytesIO(json_data)
        pythondata = JSONParser().parse(stream)
        print("python data is: ",pythondata)
        serialized_data = ServiceSerializer(data = pythondata)

        if serialized_data.is_valid():
            serialized_data.save()
            job_name = pythondata['name']
            host = pythondata['host']
            port = pythondata['port']
            query = pythondata['metric_name']
            q = 'http_server_requests_seconds_count{exception="None",instance="localhost:8080",job="Data Processor",method="GET",outcome="SUCCESS",status="200",uri="/message"}[3h:1m]'
            t1 = metric_prediction(query,q,"pcpu")
            t1.run()
            res = {'msg':"data Created"}
            res_json_data = JSONRenderer().render(res)
            return HttpResponse(res_json_data,content_type='application/json')
        res_json_data = JSONRenderer().render(serialized_data.errors)
        return HttpResponse(res_json_data,content_type='application/json')



def Allservice(request):
    if request.method == 'GET':
        allservice  = Service.objects.all()
        all_service_serializer = ServiceSerializer(allservice,many=True)
        all_service_json_data = JSONRenderer().render(all_service_serializer.data)
        return HttpResponse(all_service_json_data,content_type='application/json')

def GetTimeInterval(request):
    if request.method == 'GET':
        t = TimeInterval.objects.get(id = 1)
        t_serializer = TimeIntervalSerializer(t)
        t_json_data = JSONRenderer().render(t_serializer.data)
        return HttpResponse(t_json_data,content_type='application/json')

@csrf_exempt
def SetTimeInterval(request):
    if request.method == 'PUT':
        id = 1
        json_data  = request.body
        stream = io.BytesIO(json_data)
        python_data = JSONParser().parse(stream)
        t = TimeInterval.objects.get(id = 1)
        t_serializer = TimeIntervalSerializer(t,data=python_data)
        if t_serializer.is_valid():
            t_serializer.save()
            msg = {'msg': "time changed successfully"}
            msg_json_data = JSONRenderer().render(msg)
            return HttpResponse(msg_json_data,content_type='application/json')
        msg = JSONRenderer().render(t_serializer.errors)
        return HttpResponse(msg,content_type='application/json')

@csrf_exempt
def RemoveService(request):
    if request.method == 'DELETE':
        json_data = request.body
        stream = io.BytesIO(json_data)
        python_data = JSONParser().parse(stream)
        service_name  = python_data['name']
        service = Service.objects.filter(name =service_name)

        from influxdb import InfluxDBClient
        client = InfluxDBClient(host='localhost', port=8086)
        client.switch_database('metric_predictions_db_2')
        x = client.query("show measurements")
        x = list(x)[0]
        l = [y['name'] for y in x]
        print("measurement list: ",l)
        print("service ", service_name)
        for name in l:
            query = f"delete from {name} where job = '{service_name}'"
            print("query is: ",query)
            client.query(query)

        service.delete()
        msg = {"msg": "service removed successfully"}
        msg_json = JSONRenderer().render(msg)
        return HttpResponse(msg_json,content_type='applicatino/json')

# def Rerun():
#     threading.Timer(10,Rerun).start()
#     print("time is: ",timeobject)
#
# Rerun()