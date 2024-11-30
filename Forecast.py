#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


tcs=pd.read_csv('TCS.csv',header=0,parse_dates=True)


# In[4]:


tcs= tcs.fillna(method='ffill')


# In[5]:


tcs['Date'] = pd.to_datetime(tcs['Date'])


# In[6]:


tcs.set_index('Date',inplace=True)


# In[7]:


df1=tcs.copy()


# In[8]:


df1=df1.drop(columns=['Open','High','Low','Adj Close','Volume'])


# In[9]:


df1=df1.asfreq('B')


# In[10]:


df1= df1.fillna(method='ffill')


# In[11]:


scaler = MinMaxScaler()
scaled_data= scaler.fit_transform(df1)


# In[12]:


# Load the Model From the File
with open('Lstm_Model.pkl', 'rb') as f:
     model = pickle.load(f)


# In[13]:


import datetime

st.set_page_config(
    page_title="TCS-Closing Prices Forecast",
    page_icon=":chart_with_upwards_trend:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Custom CSS for title styling
st.markdown(
    """
    <style>
        .title {
            font-size: 36px;
            color: #003366;
            text-decoration: underline;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for background color
st.markdown(
    """
    <style>
        body {
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('TCS-Closing Prices Forecast')
st.write('Perform a Forecast for a Maximum of 30 Days')

num_days_to_forecast = st.slider('Select Number of Days to Forecast', 1, 30, 7)

if st.button('Perform Forecast'):
    if num_days_to_forecast <= 0:
        st.error('Please Select a Valid Number of Days to Forecast (Greater than 0)')
    else:
        
        forecast_input = scaled_data[-50:].reshape(1, 50, 1) 
        predictions_next_days = []

        for _ in range(num_days_to_forecast):
            # Predict the Next Data Point
            next_prediction = model.predict(forecast_input)
            predictions_next_days.append(next_prediction[0][0])

            forecast_input = np.append(forecast_input[:, 1:, :], next_prediction.reshape(1, 1, 1), axis=1)

        last_date = df1.index[-1]
        forecast_dates = [last_date + datetime.timedelta(days=i) for i in range(1, num_days_to_forecast + 1)]

        forecast_df = pd.DataFrame(predictions_next_days, index=forecast_dates, columns=['Forecast'])
        forecast_df['Forecast'] = scaler.inverse_transform(forecast_df[['Forecast']])

        st.write('Forecast Result:')
        st.write(forecast_df)

