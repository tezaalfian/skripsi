"""
URL configuration for skripsiweb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
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
from django.urls import path
from projects import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.result, name="result"),
    path("import", views.importExcel, name="importExcel"),
    path("input", views.inputData, name="inputData"),
    path("list", views.listData, name="listData"),
    path("delete/<int:id>", views.deleteData, name="deleteData"),
]
