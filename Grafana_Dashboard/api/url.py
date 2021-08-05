from django.contrib import admin
from django.urls import path,include
from . import views

urlpatterns = [
    path('/',views.home,name="homepage"),
    path('/addservice',views.AddService,name="addservice"),
    path('/allservices',views.Allservice,name="allservice"),
    path('/gettimeinterval',views.GetTimeInterval,name="gettimeinterval"),
    path('/settimeinterval',views.SetTimeInterval, name="settimeinterval"),
    path('/removeservice',views.RemoveService,name="removeservice"),
]