import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
import yfinance as yfin
from datetime import date
from datetime import timedelta


# start = '2005-01-01'
# end = '2023-02-24'

#start = '2013-01-01'
#end = '2023-02-24'


# Create a date range picker widget with default values set to today's date and 7 days from now
#start_date = st.date_input("End date", date.today() - timedelta(days=300))
#end_date = st.date_input("Start date", date.today())
# end_date = st.date_input("End date", date.today() - timedelta(days=300))

# Print the selected start and end dates
#st.write("Start date:", start_date)
#st.write("End date:", end_date)


yfin.pdr_override()


st.title('Online Automated Stock Trend Predictor')

# Create a date range picker widget with default values set to today's date and 7 days from now
start_date = st.date_input("Start date", date.today() - timedelta(days=300))
end_date = st.date_input("End date", date.today())
# end_date = st.date_input("End date", date.today() - timedelta(days=300))

# Print the selected start and end dates
st.write("Start date:", start_date)
st.write("End date:", end_date)

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
# df = data.DataReader(user_input,'yahoo',start,end)
df = data.DataReader(user_input, start_date, end_date)



latest_data = yfin.Ticker(user_input).history(period='1d')
st.subheader("Latest Price")
st.write(latest_data)


#Describing Data
st.subheader('Data from 2013 - 2023')
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)



st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


# Splitting data

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

print(data_training.shape)
print(data_testing.shape)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training)




# load my model
model = load_model('./keras_model.h5')

# Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scalar.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor



# Final Graph
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


#latest_data = yfin.Ticker(user_input).history(period='1d')

# Display the historical and predicted stock prices using a Streamlit app
#st.title("Stock Price Prediction for {}".format(user_input))
##st.line_chart(df)
#st.subheader("Latest Price")
#st.write(latest_data)
