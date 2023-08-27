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
import statsmodels.api as sm

def week_number_of_month(date_value):
    return (date_value.isocalendar()[1] - date_value.replace(day=1).isocalendar()[1] + 1)

def result(request):
    df = pd.DataFrame(list(Claim.objects.all().order_by('date').values('date', 'amount')))
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')
    df = df.resample('W-MON').sum()
    # print(df.tail(10))
    # return 0;
    # df_for_score = df.replace(0, np.nan).interpolate()
    # ARIMA
    df_fc = df[-4:]
    df = df[:-4]
    df_train = df[:-8]
    df_test = df[-8:]
    arima_params = (4, 0, 5)
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
    arima_forecast = pd.concat([arima_test[-1:], arima_forecast])

    # SES
    alpha = 0.43
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
    # ses_forecast = ses_result.forecast(steps=4)
    ses_fc = ses_result.forecast(steps=1)
    ses_forecast = pd.DataFrame({
        "amount": int(ses_fc[0])
    }, index = ses_fc.index)
    ses_fc = int(ses_fc[0])
    for index, row in df_fc.iloc[:-1].iterrows():
        ses_fc = (alpha * row['amount']) + ((1 - alpha) * ses_fc)
        next_index = index + pd.DateOffset(weeks=1)
        new = pd.DataFrame({
            "amount": int(ses_fc)
        }, index = [next_index])
        ses_forecast = pd.concat([ses_forecast, new])
    ses_forecast = pd.concat([ses_test[-1:], ses_forecast])
    # ploting arima test_forecast
    buffer = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.plot(df_test, label="Aktual")
    plt.plot(arima_test, label="Forecast ARIMA")
    plt.plot(ses_test, label="Forecast SES")
    plt.plot(arima_forecast, label="Forecast ARIMA 4 minggu ke depan")
    plt.plot(ses_forecast, label="Forecast SES 4 minggu ke depan")
    plt.ylim(0, max(df_test['amount']) + 1000000)
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
        # if mape > 1 then invalid
        mape_arima = round(mean_absolute_percentage_error([row['amount']], [arima_test.loc[index]['amount']]), 3)
        mape_ses = round(mean_absolute_percentage_error([row['amount']], [ses_test.loc[index]['amount']]), 3)
        if mape_arima > 1 and mape_ses > 1:
            mape_arima = 'Invalid'
            mape_ses = 'Invalid'
        data_test.append({
            "date": index.strftime("%b %Y") + ' Minggu ke-' + str(week_number_of_month(index)),
            "amount": row['amount'],
            "arima": arima_test.loc[index]['amount'],
            "ses": ses_test.loc[index]['amount'],
            "mape_arima": mape_arima,
            "mape_ses": mape_ses,
            "abs_arima": abs(row['amount'] - arima_test.loc[index]['amount']),
            "abs_ses": abs(row['amount'] - ses_test.loc[index]['amount'])
        })
    data_forecast = []
    for index, row in arima_forecast[1:].iterrows():
        data_forecast.append({
            "date": index.strftime("%b %Y") + ' Minggu ke-' + str(week_number_of_month(index)),
            "arima": row['amount'],
            "ses": ses_forecast.loc[index]['amount']
        })
    if arima_mape > 1 and ses_mape > 1:
        arima_mape = 'Invalid'
        ses_mape = 'Invalid'
    else:
        arima_mape = round(arima_mape * 100, 1)
        ses_mape = round(ses_mape * 100, 1)
    return render(request, "result.html", {
        "graph_test": graph_test,
        "arima_mape": arima_mape,
        "ses_mape": ses_mape,
        'arima_forecast': arima_forecast,
        'ses_forecast': ses_forecast,
        'data_test': data_test,
        'data_forecast': data_forecast
    })

def importExcel(request):
    if request.method == "POST":
        # print(request.FILES["dataset"])
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

def prediksi(request):
    if request.GET.get("jumlah") and request.GET.get("p") and request.GET.get("alpha"):
        arima_params = (int(request.GET.get("p")), int(request.GET.get("d")), int(request.GET.get("q")))
        count = int(request.GET.get("jumlah"))
        df = pd.DataFrame(list(Claim.objects.all().order_by('date').values('date', 'amount')))
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        df = df.set_index('date')
        df = df.resample('W-MON').sum()
        # extract last 4 week
        df_fc = df[-count:]
        df = df[:-count]
        # ARIMA
        arima_model = ARIMA(df, order=arima_params)
        arima_train = arima_model.fit()
        arima_fc = arima_train.forecast(steps=count)
        arima_fc = arima_fc.rename('amount').astype(int).to_frame()
        # SES
        alpha = float(request.GET.get("alpha"))
        ses_model = SimpleExpSmoothing(df)
        ses_result = ses_model.fit(smoothing_level=alpha)
        ses_fc = ses_result.forecast(steps=1)
        ses_forecast = pd.DataFrame({
            "amount": int(ses_fc[0])
        }, index = ses_fc.index)
        ses_fc = int(ses_fc[0])
        for index, row in df_fc.iloc[:-1].iterrows():
            ses_fc = (alpha * row['amount']) + ((1 - alpha) * ses_fc)
            next_index = index + pd.DateOffset(weeks=1)
            new = pd.DataFrame({
                "amount": int(ses_fc)
            }, index = [next_index])
            ses_forecast = pd.concat([ses_forecast, new])
        # ploting arima_fc
        buffer = BytesIO()
        plt.figure(figsize=(12, 6))
        plt.plot(arima_fc, label="Forecast ARIMA")
        plt.plot(ses_forecast, label="Forecast SES")
        plt.title('Forecast')
        plt.xlabel('Tanggal/Minggu')
        plt.ylabel('Besar Klaim')
        plt.legend()
        plt.savefig(buffer, format='png')
        plt.clf()
        buffer.seek(0)
        image_png = buffer.getvalue()
        graph_fc = base64.b64encode(image_png)
        graph_fc = graph_fc.decode('utf-8')
        buffer.close()
        data_fc = []
        for index, row in arima_fc.iterrows():
            data_fc.append({
                "date": index.strftime("%b %Y") + ' Minggu ke-' + str(week_number_of_month(index)),
                "arima": row['amount'],
                "ses": ses_forecast.loc[index]['amount']
            })
        return render(request, "prediksi.html", {
            "graph_fc": graph_fc,
            "data_fc": data_fc
        })
    return render(request, "prediksi.html")