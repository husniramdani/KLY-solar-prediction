from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import numpy as np
import re

import datetime
import time

st.write("""
# Solar Prediction
""")

# ================== DATA PREPARATION ==================
df = pd.read_csv("./SolarPrediction.csv")
# st.write(df)

# drop data UNIXTIME
# df = df.drop('UNIXTime', axis=1)
# st.write(df)

# drop data "Data"
df = df.drop('Data', axis=1)
# st.write(df)

# drop data Time
df = df.drop('Time', axis=1)
# st.write(df)

df['SunriseHour'] = df['TimeSunRise'].apply(
    lambda x: re.search(r'^\d+', x).group(0)).astype(np.int)
df['SunriseMinute'] = df['TimeSunRise'].apply(
    lambda x: re.search(r'(?<=:)\d+(?=:)', x).group(0)).astype(np.int)

df['SunsetHour'] = df['TimeSunSet'].apply(
    lambda x: re.search(r'^\d+', x).group(0)).astype(np.int)
df['SunsetMinute'] = df['TimeSunSet'].apply(
    lambda x: re.search(r'(?<=:)\d+(?=:)', x).group(0)).astype(np.int)

df = df.drop(['TimeSunRise', 'TimeSunSet'], axis=1)
df['SunriseHour'].unique()
df = df.drop(['SunriseHour'], axis=1)

st.write(df)

# ================== START TO SPLITTING DATA ==================

y = df['Radiation']
X = df.drop('Radiation', axis=1)

X.info()

# print('shape of y', y.shape)
# print('shape of x', X.shape)

# mengimport fungsi untuk membagi data menjadi data training dan data testing
# from sklearn.metrics import mean_squared_error, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=300)


# input data test
st.sidebar.header('Input Parameters')


def user_input_features():
    tgl = st.sidebar.date_input("Tanggal", datetime.date(2016, 9, 1))

    temperature = st.sidebar.slider('Temperature', 0, 100, 51)
    pressure = st.sidebar.slider('Pressure', 20.00, 50.00, 30.43)
    humidity = st.sidebar.slider('Humidity', 0, 200, 103)
    wind_direction = st.sidebar.slider('Wind Direction', 0.00, 400.00, 77.27)
    speed = st.sidebar.slider('Speed', 0.00, 50.00, 11.25)

    sunsetHour = st.sidebar.slider('Sunset Hour', 0, 24, 18)
    sunsetMinute = st.sidebar.slider('Sunset Minute', 0, 60, 38)
    sunriseMinute = st.sidebar.slider('Sunrise Minute', 0, 60, 7)

    date_format = int(time.mktime(tgl.timetuple()))
    data = {
        'UNIXTime': date_format,
        'Temperature': temperature,
        'Pressure': pressure,
        'Humidity': humidity,
        'WindDirection(Degrees)': wind_direction,
        'Speed': speed,
        'SunriseMinute': sunsetMinute,
        'SunsetHour': sunsetHour,
        'SunsetMinute': sunriseMinute,
    }
    features = pd.DataFrame(data, index=[0])
    return features


data_test = user_input_features()

st.write('Data Test :', data_test)

# ================== Randdom Forest ==================
st.subheader('Predict Using Random Forest')
from sklearn.ensemble import RandomForestRegressor

# melatih model menggunakan parameter random_state
rf_reg = RandomForestRegressor(max_depth=25, n_estimators=100, random_state=19)
rf_reg.fit(X_train, y_train)

# Mean R2 Score Training
# st.write("Mean R2 Score Training : ", rf_reg.score(X_train, y_train))

# Mean R2 Score Testing
# st.write("Mean R2 Score Testing : ", rf_reg.score(X_test,y_test))

# Prediksi menggunakan data testing
rf_pred = rf_reg.predict(data_test)
st.write("Hasil Prediksi : ", rf_pred[0])
