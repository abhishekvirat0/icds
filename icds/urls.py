"""icds URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
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
from django.conf import settings
from django.conf.urls import url
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from foodIcds import views

urlpatterns = [
                  path('admin/', admin.site.urls),
                  path('', views.Index.as_view(), name='home'),
                  path('category/', views.Category.as_view(), name='category'),
                  path('foodSelection/', views.FoodSelection.as_view(), name='FoodSelection'),
                  path('anganwadi/foodSelection/', views.FoodSelection.as_view(), name='Anganwadi_FoodSelection'),
                  url(r'^foodCost/$', views.FoodCost.as_view(), name='FoodCost'),
                  path('result/', views.Result.as_view(), name='result'),
                  path('veg/result/', views.VegResult.as_view(), name='veg_result'),
                  path('veg/select/', views.Recipe.as_view(), name='recipe'),
                  path('veg/quantity/', views.VegQuantity.as_view(), name='vegQuantity'),
                  path('filter-data', views.filter_data, name='filter_data'),
                  path('filter-vegetable-data', views.filter_vegetable_data, name='filter_vegetable_data'),
                  path('result/pdf/', views.GetPdf.as_view(), name='pdf'),
              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
