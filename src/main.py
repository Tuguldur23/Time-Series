import torch.optim as optim
import torch.nn as nn
from optimize import Optimization
from model_loader import build_model
import yaml
from data_loader import data_loader, feature_label_split, get_data
import pandas as pd
import streamlit as st
from data import format_predictions, calculate_metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from plotly import graph_objs as go

st.title("Таамаглал")
yes = ['Test', 'Predict']
hey = st.selectbox('Сонголт', yes)
years = [2017, 2018, 2019, 2020, 2021, 2022]
year = st.selectbox("Жилээ сонгох", years)

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sel_months = [12]
select_months = [1, 2, 3, 4, 5, 6]
if year == 2017:
    month = st.selectbox("Сараа сонгох", sel_months)
elif year == 2022:
    month = st.selectbox("Сараа сонгох", select_months)
else:
    month = st.selectbox('Сараа сонгох', months)

day = st.text_input("Өдөрөө сонгох")

models = ['lstm', 'rnn', 'gru']
select_model = st.selectbox("Загвар сонгох", models)
ok = st.button("Таамаглал гаргах")

config = yaml.safe_load(open("/home/tuugu/Documents/Time-Series/configs/config.yaml"))
if ok:
    st.title('Loading...')
    scaler = MinMaxScaler()
    df = pd.read_csv(config['data'], parse_dates=True, sep=',', index_col=['Datetime'])
    df = df[['Year', 'Month', 'Hour', 'Temperature', 'Prev_day', 'Prev_15min', 'Prev_hour', 'Relative Humidity', 'Power']]
    df = get_data(year, month, day, df)
    X, y = feature_label_split(df, config['target_col'])
    X_test_arr = scaler.fit_transform(X)
    y_test_arr = scaler.fit_transform(y)
    test_loader_one = data_loader(X_test_arr, y_test_arr)
    model = build_model(select_model, config)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=float(config['train_params']['learning_rate']),
                       weight_decay=float(config['train_params']['weight_decay']))
    opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
    predictions, values = opt.evaluate(test_loader_one, batch_size=1, n_features=config['model_params']['input_dim'])
    df_result = format_predictions(predictions, values, X, scaler)
    st.title('Done!')
    value = df_result["value"].to_list()
    pred = df_result["prediction"].to_list()
    if hey == 'Test':
        plt.rcParams["figure.autolayout"] = True
        plt.plot(value, label="Бодит утга")
        plt.plot(pred, label="Таамагласан утга")
        plt.legend(loc="upper right", prop={"size":15})
        plt.xticks([0, 35, 66, 95], [f"{year}-{month}-{day}-0:00", f"{year}-{month}-{day}-8:30", f"{year}-{month}-{day}-16:30", f"{year}-{month}-{day}-23:45"])
        result_metrics = calculate_metrics(df_result)
        pred_max = df_result['prediction'].sum()
        value_max = df_result['value'].sum()
        st.title("Таамагласан өдрийн хүснэгтэн үр дүн")
        st.write(f"Тухайн өдрийн таамагласан нийт чадал {pred_max} Ватт")
        st.write(f"Тухайн өдрийн нийт чадал {value_max} Ватт")
        st.write(f"Тухайн өдрийн таамагласан хамгийн их чадал {df_result['prediction'].max()} Ватт")
        st.write(f"Тухайн өдрийн хамгийн их чадал {df_result['value'].max()} Ватт")
        st.write(f"Таамагласан ${(pred_max/1000)*0.12} доллар")
        st.write(f"Таамагласан {((pred_max/1000)*0.12)*3120} төгрөг")
        st.write(f"Бодит ${(value_max / 1000)*0.12} доллар")
        st.write(f"Бодит {((value_max/1000)*0.12)*3120} төгрөг")
        st.dataframe(data=df_result)
        st.download_button('Татах CSV', df_result.to_csv(), file_name=f"{year}{month}{day}.csv")
        st.title("Үр дүнгийн утгууд")
        st.write(result_metrics)
        st.title("Таамагласан өдрийн зурган үр дүн")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['prediction'], name='Таамаглал'))
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['value'], name='Бодит утга'))
        fig.layout.update(xaxis_title="Хугацаа", yaxis_title="Чадал", xaxis_rangeslider_visible=True, font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"))
        st.plotly_chart(fig)
    else:
        plt.rcParams["figure.autolayout"] = True
        plt.plot(pred, label="Таамагласан утга")
        plt.legend(loc="upper right", prop={"size":15})
        plt.xticks([0, 35, 66, 95], [f"{year}-{month}-{day}-0:00", f"{year}-{month}-{day}-8:30", f"{year}-{month}-{day}-16:30", f"{year}-{month}-{day}-23:45"])
        result_metrics = calculate_metrics(df_result)
        pred_max = df_result['prediction'].sum()
        st.title("Таамагласан өдрийн үр дүн")
        st.write(f"Тухайн өдрийн таамагласан нийт чадал {pred_max} Ватт")
        st.write(f"Тухайн өдрийн таамагласан хамгийн их чадал {df_result['prediction'].max()} Ватт")
        st.write(f"Таамагласан ${(pred_max/1000)*0.12} доллар")
        st.write(f"Таамагласан {((pred_max/1000)*0.12)*3120} төгрөг")
        st.dataframe(data=df_result['prediction'])
        if len(day) == 1 and len(str(month)) == 1:
            st.download_button('Татах CSV', df_result['prediction'].to_csv(), file_name=f"{year}0{month}0{day}.csv")
        elif len(day) == 1 and len(str(month)) == 2:
            st.download_button('Татах CSV', df_result['prediction'].to_csv(), file_name=f"{year}{month}0{day}.csv")
        elif len(day) == 2 and len(str(month)) == 1:
            st.download_button('Татах CSV', df_result['prediction'].to_csv(), file_name=f"{year}0{month}{day}.csv")
        else:
            st.download_button('Татах CSV', df_result['prediction'].to_csv(), file_name=f"{year}{month}{day}.csv")
        st.title("Таамагласан өдрийн зурган үр дүн")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_result.index, y=df_result['prediction'], name='Таамаглал'))
        fig.layout.update(xaxis_title="Хугацаа", yaxis_title="Чадал", xaxis_rangeslider_visible=True, font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"))
        st.plotly_chart(fig)
    if select_model == "lstm":
        st.title("LSTM тест хариу")
        st.dataframe(data=pd.read_csv("/home/tuugu/Documents/Time-Series/datas/uzuuleh.csv"))
    elif select_model == "gru":
        st.title("GRU тест хариу")
        st.dataframe(data=pd.read_csv("/home/tuugu/Documents/Time-Series/datas/uzuuleh_gru.csv"))
    else:
        st.title("RNN тест хариу")
        st.dataframe(data=pd.read_csv("/home/tuugu/Documents/Time-Series/datas/uzuuleh_rnn.csv"))
