import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
st.title('🏠House Price prediction using ML')
st.image('https://cdn.dribbble.com/userupload/23217976/file/original-203940eb89dec42ee25f6af662af24dc.gif')
df = pd.read.csv('house_data_csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('Select House features')
st.sidebar.image('https://cdn.dribbble.com/userupload/23217976/file/original-203940eb89dec42ee25f6af662af24dc.gif')
all_value = []
for i in X:

 min_value = int(X[i].min())
 max_value = int(y[i].max())
 ans =  st.sidebar.slider(f'Select {i} value', 'min_value','max_value')
 all_value.append(ans)

# st.write(all_value)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

final_value = scaler/transform([all_value])

model = RandomForestRegressor()
model.fit(X,y)
house_price = model.predict(final_value)

with st.spinner('Predicting House Price'):
 time.sleep(1)
 msg = f'''House Price is ${house_price*100000}'''
 st.success(msg)

st.markdown{'''**Design And Developed by : Amit Pratap Singh**'''}




