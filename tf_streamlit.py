import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras import models
import joblib

st.title("Used car price prediction app")
st.subheader("""This app uses a pretrained CNN model to predict a used car's potential price """) 
st.write("""Specify your parameters and hit "Ready" button,then press "Predict Price" to get the predicted value """)
@st.cache
def load_data(allow_output_mutation=True):
    return pd.read_csv('car_data.csv')
Data = load_data()
x =Data[["Kilometers_Driven","Mileage","Engine","Power","Age"]] #independent
y = Data["Price"] #dependent

Data = Data.drop(Data[(Data.Power <300.0 ) & (Data.Price > 120000)].index)
Data = Data.drop(Data[(Data.Power >300.0 ) & (Data.Price < 40000)].index)
Data = Data.drop(Data[(Data.Power >350.0 ) & (Data.Price < 60000)].index)
Data = Data.drop('Seats',axis=1)
Data = Data.drop('Owner_Type',axis=1)
Data = Data.drop('Fuel_Type',axis=1)
Data = Data.drop('Transmission',axis=1)



st.subheader(""" This plot represents how much each value affects the  car's price """)
top_features = Data.corr()[['Price']].sort_values(by=['Price'],ascending=False).head(10)
fig, ax = plt.subplots()
sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=-1, ax=ax)
st.write(fig)
def create_df(Kilometers_Driven,Mileage,Engine,Power,Age):
    X_scaled =['Kilometers_Driven','Mileage','Engine','Power','Age']
    X_scale = pd.DataFrame(columns=X_scaled)
    values_to_add = {'Kilometers_Driven':Kilometers_Driven, 'Mileage':Mileage,'Engine':Engine,'Power':Power,'Age':Age}
    row_to_add = pd.Series(values_to_add, name='0')
    X_scale = X_scale.append(row_to_add)
    
    
    return X_scale

# Define components for the sidebar
st.sidebar.header('Input Features')
    
def main():    
        Kilometers_Driven = st.sidebar.slider(
            label='Kilometers_Driven',
            min_value=float(0),
            max_value=float(800000.0),
            value=float(),
            step=500.0)
        Mileage = st.sidebar.slider(
            label='Mileage',
            min_value=float(0.0),
            max_value=float(35),
            value=float(),
            step=0.1)
        Engine = st.sidebar.slider(
            label='Engine',
            min_value=float(600),
            max_value=float(5500),
            value=float(),
            step=10.0)
        Power = st.sidebar.slider(
            label='Engine power',
            min_value=float(0.0),
            max_value=float(560.0),
            value=float(),
            step=1.0)
        Age = st.sidebar.slider(
            label='Age',
            min_value=float(1),
            max_value=float(25),
            value=float(),
            step=1.0)
        if st.button("Ready"):
            st.header ('Specified parameters')
            st.write(Kilometers_Driven,Mileage,Engine,Power,Age)
            
        if st.button("Predict_Price"):
        
            model = tf.keras.models.load_model('my_model_1.hdf5')
            s_scaler = joblib.load("scaler.save")
            X_sc = create_df(Kilometers_Driven,Mileage,Engine,Power,Age)
            
            s = np.load('std.npy')
            m = np.load('mean.npy')
            X_sc = (X_sc - m) /s
            
            y_pred = model.predict(X_sc)
            st.header('Predicted Car Price')
            st.write(y_pred)
        

        
    
    
if __name__ == "__main__":
    main()
   
