import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
st.title('🏠 House Price prediction using ML')
st.image('https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/0532e265355325.5af1a031e198f.gif')

df = pd.read_csv('house_data.csv')
X = df.iloc[:,:-3]
y = df.iloc[:,-1]

st.sidebar.title('🏠 Select House features')
st.sidebar.image('https://img.freepik.com/premium-photo/hightech-modular-housing-with-customizable-smart-home-features-solid-color-background-4k-ultra-hd_964851-116824.jpg')
all_value = []
for i in X:
  min_value = int(X[i].min())
  max_value = int(X[i].max())
  ans = st.sidebar.slider(f'Select {i} value',min_value, max_value)
  all_value.append(ans)

# st.write(all_value)

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
final_value = scaler.transform([all_value])

@st.cache_data
def model_run(X,y):
  model = RandomForestRegressor()
  model.fit(X,y)
  return model

model = model_run(X,y)  
house_price = model.predict(final_value)[0]

with st.spinner('Predicting House price'):
  time.sleep(1)
msg = f'''House price is: $ {round(house_price*100000,2)}'''
st.success(msg)

st.markdown('''**Design and Developed by: Ankush verma**''')






