from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .models import Claim
from django.shortcuts import redirect
from .bulk_create_manager import BulkCreateManager
from django.contrib import messages

# Create your views here.
def result(request):
    return render(request, "base.html")

def importExcel(request):
    if request.method == "POST":
        print(request.FILES["dataset"])
        df = pd.read_excel(request.FILES['dataset'])
        df = df[['TANGGAL MASUK','TOTAL APPROVE']].dropna()
        df['TANGGAL MASUK'] = pd.to_datetime(df['TANGGAL MASUK'], format='%Y-%m-%d')
        df = df.astype({'TOTAL APPROVE':'int'})
        df = df[(df[['TOTAL APPROVE']] != 0).all(axis=1)]
        df = df.sort_values(by=['TANGGAL MASUK'])
        bulk_mgr = BulkCreateManager(chunk_size=100)
        for index, row in df.iterrows():
            bulk_mgr.add(Claim(date=row['TANGGAL MASUK'], amount=row['TOTAL APPROVE']))
        bulk_mgr.done()
        messages.success(request, "Data berhasil diimport")
        return redirect('/import')

    return render(request, "import.html")

def inputData(request):
    return HttpResponse("Hello, world. You're at the polls index.")