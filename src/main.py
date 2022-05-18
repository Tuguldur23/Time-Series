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

st.title("Таамаглал")

years = [2017, 2018, 2019, 2020]
year = st.selectbox("Жилээ сонгох", years)

months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
month = st.selectbox("Сараа сонгох", months)

day = st.text_input("Өдөрөө сонгох")

models = ['lstm', 'rnn', 'gru']
select_model = st.selectbox("Загвар сонгох", models)
ok = st.button("Таамаглал гаргах")

config = yaml.safe_load(open("/home/tuugu/Documents/Time-Series/configs/config.yaml"))
if ok:
    scaler = MinMaxScaler()
    df = pd.read_csv(config['data'], parse_dates=True, sep=',')
    df = df[['Datetime', 'Year', 'Month', 'Hour', 'Temperature', 'Prev_day', 'Prev_15min', 'Prev_hour', 'Relative Humidity', 'Power']]
    df = get_data(year, month, day, df)
    df = df[['Year', 'Month', 'Hour', 'Temperature', 'Prev_day', 'Prev_15min', 'Prev_hour', 'Relative Humidity', 'Power']]
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
    value = df_result["value"].to_list()
    pred = df_result["prediction"].to_list()
    plt.rcParams["figure.autolayout"] = True
    plt.plot(value, label="Бодит утга")
    plt.plot(pred, label="Таамагласан утга")
    plt.legend(loc="upper right", prop={"size":15})
    plt.xticks([25, 65], [f"{year}-{month}-{day}-8:00", f"{year}-{month}-{day}-19:00"])
    result_metrics = calculate_metrics(df_result)
    st.dataframe(data=df_result)
    st.write(result_metrics)
    st.pyplot(fig=plt)
