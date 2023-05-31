from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
from .models import Claim
from django.shortcuts import redirect
from .bulk_create_manager import BulkCreateManager
from django.contrib import messages
import numpy as np
from statsmodels.tsa.api import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

def week_number_of_month(date_value):
    return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)

def result(request):
    df = pd.DataFrame(list(Claim.objects.all().order_by('date').values('date', 'amount')))
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    df = df.resample('W-MON').sum()
    # df_for_score = df.replace(0, np.nan).interpolate()
    # ARIMA
    df_train = df[:-8]
    df_test = df[-8:]
    arima_params = (3, 2, 0)
    arima_model = ARIMA(df_train, order=arima_params)
    arima_train = arima_model.fit()
    arima_test = arima_train.forecast(steps=len(df_test))
    arima_mape = mean_absolute_percentage_error(df_test, arima_test)
    arima_test = arima_test.rename('amount').astype(int).to_frame()
    # forecast real data
    arima_model = ARIMA(df, order=arima_params)
    arima_result = arima_model.fit()
    arima_forecast = arima_result.forecast(steps=4)
    arima_forecast = arima_forecast.rename('amount').astype(int).to_frame()

    # SES
    alpha = 0.35
    ses_test = pd.DataFrame()
    for i in range(len(df_test), 0, -1):
        model = SimpleExpSmoothing(df[:-i])
        results = model.fit(smoothing_level=alpha)
        next = results.forecast(steps=1)
        new = pd.DataFrame({
            "amount": int(next[0])
        }, index = next.index)
        ses_test = pd.concat([ses_test, new])
    ses_mape = mean_absolute_percentage_error(df_test, ses_test)
    # forecast real data
    ses_model = SimpleExpSmoothing(df)
    ses_result = ses_model.fit(smoothing_level=alpha)
    ses_forecast = ses_result.forecast(steps=4)
    ses_forecast = ses_forecast.rename('amount').astype(int).to_frame()
    # ploting arima test_forecast
    buffer = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(df_test, label="Aktual")
    plt.plot(arima_test, label="Forecast ARIMA")
    plt.plot(ses_test, label="Forecast SES")
    plt.plot(arima_forecast, label="Forecast ARIMA 4 minggu ke depan")
    plt.plot(ses_forecast, label="Forecast SES 4 minggu ke depan")
    plt.title('Actual vs Forecast')
    plt.xlabel('Tanggal/Minggu')
    plt.ylabel('Besar Klaim')
    plt.legend()
    plt.savefig(buffer, format='png')
    plt.clf()
    buffer.seek(0)
    image_png = buffer.getvalue()
    graph_test = base64.b64encode(image_png)
    graph_test = graph_test.decode('utf-8')
    buffer.close()
    data_test = []
    for index, row in df_test.iterrows():
        # create object date, amount, forecast arima & ses
        data_test.append({
            "date": index.strftime("%b %Y") + ' Minggu ke-' + str(week_number_of_month(index)),
            "amount": row['amount'],
            "arima": arima_test.loc[index]['amount'],
            "ses": ses_test.loc[index]['amount'],
            "mape_arima": round(mean_absolute_percentage_error([row['amount']], [arima_test.loc[index]['amount']]), 3),
            "mape_ses": round(mean_absolute_percentage_error([row['amount']], [ses_test.loc[index]['amount']]), 3)
        })
    data_forecast = []
    for index, row in arima_forecast.iterrows():
        data_forecast.append({
            "date": index.strftime("%b %Y") + ' Minggu ke-' + str(week_number_of_month(index)),
            "arima": row['amount'],
            "ses": ses_forecast.loc[index]['amount']
        })
    return render(request, "result.html", {
        "graph_test": graph_test,
        "arima_mape": round(arima_mape * 100, 2),
        "ses_mape": round(ses_mape * 100, 2),
        'arima_forecast': arima_forecast,
        'ses_forecast': ses_forecast,
        'data_test': data_test,
        'data_forecast': data_forecast
    })

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
    if request.method == "POST":
        date = request.POST['date']
        amount = request.POST['amount']
        Claim.objects.create(date=date, amount=amount)
        messages.success(request, "Data berhasil ditambahkan")
        return redirect('/input')
    return render(request, "input.html")

def listData(request):
    data = Claim.objects.all().order_by('-id')
    return render(request, "list.html", {
        "datas": data
    })

def deleteData(request, id):
    Claim.objects.filter(id=id).delete()
    messages.success(request, "Data berhasil dihapus")
    return redirect('/list')