import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tensorflow
from tensorflow import keras
from keras import models

st.write("""
         # Car Price Prediction:# """)
st.write(""" ## Data analysis & CNN model construction ## """)
#@st.cache(allow_output_mutation=True)
def load_data():
    Data = pd.read_csv('car_data.csv')
    return Data

Data = load_data()
st.header("Input Data (top 10 rows)")
st.write(Data.head(10))
st.write('---')
st.write(""" # Ploting data significance #
## A heatmap is initially constructed to demonstrate the most significant corellations between the data ## """)
st.write(""" ### The top five attributes that influence Price are :Power,Engine,Milage,Age and Kilometers_Driven ### """)


Data.Transmission = Data.Transmission.map({'Manual':0,'Automatic':1})
Data.Fuel_Type= Data.Fuel_Type.map({'Petrol':0,'Diesel':1,'CNG':2,'LPG':3})


fig, ax = plt.subplots()
heat=sns.heatmap(Data.corr(),annot=True,ax=ax)
st.write(fig)

st.write("---")
st.write(""" ## Visualising parameters with respect to price ## """) 
st.write(""" ### Data with weak influence over Price ### """)
fig = plt.figure(figsize=(15,12))
fig.add_subplot(3,2,1)
sns.scatterplot(x=Data.Owner_Type, y =Data.Price)
fig.add_subplot(3,2,2)
sns.scatterplot(x=Data.Fuel_Type, y =Data.Price)
fig.add_subplot(3,2,3)
sns.scatterplot(x=Data.Seats, y =Data.Price)
fig.add_subplot(3,2,4)
sns.scatterplot(x=Data.Fuel_Type, y =Data.Price)
fig.add_subplot(3,2,5)
sns.scatterplot(x=Data.Transmission, y =Data.Price)
st.write(fig)
st.write('Data with weak influence tend to have arbitraty values as the price increases ')

st.write(""" ### Data with strong influence over Price ### """)
fig = plt.figure(figsize=(16,10))
fig.add_subplot(2,2,1)
sns.scatterplot(x=Data.Engine,y =Data.Price)
fig.add_subplot(2,2,2)
sns.scatterplot(x=Data.Power,y =Data.Price)
fig.add_subplot(2,2,3)
sns.scatterplot(x=Data.Age,y =Data.Price)
fig.add_subplot(2,2,4)
sns.scatterplot(x=Data.Mileage,y =Data.Price)
st.write(fig)
st.write('Data with strong influence tend to have a liner correllation as the price increases ')
st.write("---")
st.write('On the previous graphs we can observe data points with values that differ greatly from the majority of our data,outliers that might intefeer with the results of our model training') 
sns.countplot(x=Data.Seats)
def plot_data(col,name ,    discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        sns.stripplot(x=col, y=Data.Price, data=Data, ax=ax[0])
        sns.countplot(col, ax=ax[1])
        title = ('Outlier Analysis' + " " +name)
        fig.suptitle(title)
        st.write(fig)
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y=Data.Price, data=Data, ax=ax[0])
        sns.distplot(col, kde=False, ax=ax[1])
        title = ('Outlier Analysis' + " " +name)
        fig.suptitle(title)
        st.write(fig)
        

      
plot_data(Data.Power,'Power')
Data = Data.drop(Data[(Data.Power <300.0 ) & (Data.Price > 120000)].index)
Data = Data.drop(Data[(Data.Power >300.0 ) & (Data.Price < 40000)].index)
Data = Data.drop(Data[(Data.Power >350.0 ) & (Data.Price < 60000)].index)

plot_data(Data.Power,'Power')
plot_data(Data.Engine,'Engine')
Data = Data.drop(Data[(Data.Engine <3100 )& (Data.Price > 80000)].index)
plot_data(Data.Engine,'Engine')


st.write("""By eliminating some of the outliers from the dataset,the unique ones with high values, we improve performance by reducing the range of possible price values """)
st.write('---')

Data = Data.drop('Name',axis=1)
Data = Data.drop('Seats',axis=1)
Data = Data.drop('Owner_Type',axis=1)
Data = Data.drop('Fuel_Type',axis=1)
Data = Data.drop('Transmission',axis=1)
st.write("---")
st.write("Training Model")
st.write("---")
X = Data.drop('Price',axis =1).values
y = Data['Price'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.fit_transform(X_test.astype(np.float))

std  = np.sqrt(s_scaler.var_)
np.save('std.npy',std )
np.save('mean.npy',s_scaler.mean_)

import joblib
scaler_filename = "scaler.save"
joblib.dump(s_scaler, scaler_filename)



# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
mse = MeanSquaredError()
e_s = EarlyStopping(monitor='val_loss', patience=30)

# having 9 neuron is based on the number of available features
model = Sequential()
model.add(Dense(320,activation='relu',input_shape=[len(X[1])]))
model.add(Dense(384,activation='relu'))
model.add(Dense(352,activation='relu'))
model.add(Dense(448,activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='Adam',loss='mse')

history = model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=64,epochs=100,verbose = 2,callbacks=[e_s])

stringlist = []
model.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)
st.header('Summary of the CNN model:')
st.text(short_model_summary)

loss_df = pd.DataFrame(model.history.history)
st.write("---")
st.write('This graph presents the training loss (Blue line) as well as the validation loss(Yellow line)')
st.line_chart(loss_df)
st.write("---")

y_pred = model.predict(X_test)
dataf = pd.DataFrame({'predicted_price': y_pred.ravel(), 'actual_price': y_test.ravel()})
#result = pd.DataFrame({'y_pred':[y_pred[1:20,:]],'y_yest':[y_test[1:20]]})


#st.write(dataf)
st.write("---")
from sklearn import metrics
st.write('Mean Absolute Error')
st.write('MAE:', metrics.mean_absolute_error(y_test, y_pred))
st.write('Mean Squared Error')
st.write('MSE:', metrics.mean_squared_error(y_test, y_pred))
st.write('Root Mean Squared Error')
st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
st.write('Total varidation Score of the model that indicates dispersion of errors of the model')
st.write('Variation Score:',metrics.explained_variance_score(y_test,y_pred))
st.write("""After handling and selecting data properly , our CNN regression model performs greatly.With just 100 itterations the mean absolute error in below 2000 euros""")



st.write("---")
st.write(' Visualizing Our predictions with the red line compared to the actual values represented with blue circles')
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
# Perfect predictions

plt.plot(y_test,y_test,'r')
st.write(fig)
st.write("---")
st.write(' 20 representative values of predicted values, actual values and the corresponding  error value of the prediction')
error = dataf.predicted_price - dataf.actual_price
dataf2 = dataf[:20]
dataf2['difference'] = error[0:20]
st.write(dataf2)

fig,ax = plt.subplots()
plt.hist(error, bins=10)
plt.xlabel('Prediction Error [MPG]')
st.pyplot(fig)
st.write('This bar charts gives us information about the distribution of price errors,very few of them exceed 10.000 euros') 
















