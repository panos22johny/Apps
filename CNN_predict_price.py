#import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model

#import csv Data file 
Data = pd.read_csv('Spotawheel_case_study (1).csv')
Data.head(5).T

#change categorical data to corresponding numbers for every category
Data.Fuel_Type= Data.Fuel_Type.map({'Petrol':0,'Diesel':1,'CNG':2,'LPG':3})
Data.Transmission = Data.Transmission.map({'Manual':0,'Automatic':1})
#get some information about our Data-Set
Data.info()
Data.describe().transpose()

#check for the best data correlation between price and the rest of the values
top_features = Data.corr()[['Price']].sort_values(by=['Price'],ascending=False).head(10)
plt.figure(figsize=(5,10))
sns.heatmap(top_features,cmap='rainbow',annot=True,annot_kws={"size": 16},vmin=-1)

###visualizing car prices and every other variable
plt.show()
def plot_data(col, discrete=False):
    if discrete:
        fig, ax = plt.subplots(1,2,figsize=(14,6))
        sns.stripplot(x=col, y=Data.Price, data=Data, ax=ax[0])
        sns.countplot(col, ax=ax[1])
        fig.suptitle(' Analysis')
    else:
        fig, ax = plt.subplots(1,2,figsize=(12,6))
        sns.scatterplot(x=col, y=Data.Price, data=Data, ax=ax[0])
        sns.distplot(col, kde=False, ax=ax[1])
        fig.suptitle('Analysis')

#Searching for outliers in the data,ploting price over all the other significant variables from the heatmap
#if a data point is devianting it is deleted       
plot_data(Data.Power)
plt.show()
Data = Data.drop(Data[(Data.Power <300.0 ) & (Data.Price > 120000)].index)
Data = Data.drop(Data[(Data.Power >300.0 ) & (Data.Price < 40000)].index)
Data = Data.drop(Data[(Data.Power >350.0 ) & (Data.Price < 60000)].index)
plot_data(Data.Engine)
plt.show()
plot_data(Data.Transmission)
plt.show()
Data = Data.drop(Data[(Data.Transmission == 1 )& (Data.Price > 110000)].index)
plot_data(Data.Power)
plt.show()
Data = Data.drop(Data[(Data.Power <300.0 ) & (Data.Price > 120000)].index)
Data = Data.drop(Data[(Data.Power >450.0 ) & (Data.Price < 120000)].index)
plot_data(Data.Kilometers_Driven)
plt.show()

#visualising range and volume of prices, ploting to see how the price deviates
fig = plt.figure(figsize=(10,10))
fig.add_subplot(2,1,1)
sns.histplot(data =Data.Price)
fig.add_subplot(2,1,2)
sns.boxplot(x = Data.Price)

#most significant data
fig = plt.figure(figsize=(16,16))
fig.add_subplot(3,2,1)
sns.scatterplot(x=Data.Kilometers_Driven, y =Data.Price)
fig.add_subplot(3,2,2)
sns.scatterplot(x=Data.Engine,y =Data.Price)
fig.add_subplot(3,2,3)
sns.scatterplot(x=Data.Mileage,y =Data.Price)
fig.add_subplot(3,2,4)
sns.scatterplot(x=Data.Power,y =Data.Price)
fig.add_subplot(3,2,5)
sns.scatterplot(x=Data.Age,y =Data.Price)
plt.show()

#less significant data
fig = plt.figure(figsize=(16,10))
fig.add_subplot(2,2,1)
sns.scatterplot(x=Data.Transmission, y =Data.Price)
fig.add_subplot(2,2,2)
sns.scatterplot(x=Data.Owner_Type,y =Data.Price)
fig.add_subplot(2,2,3)
sns.scatterplot(x=Data.Fuel_Type,y =Data.Price)
fig.add_subplot(2,2,4)
sns.scatterplot(x=Data.Seats,y =Data.Price)
plt.show()

# drop some unnecessary columns with poor coorelation to Price
Data = Data.drop('Name',axis=1)
Data = Data.drop('Seats',axis=1)
Data = Data.drop('Owner_Type',axis=1)
Data = Data.drop('Fuel_Type',axis=1)

#prepare data for training
X = Data.drop('Price',axis =1).values
y = Data['Price'].values
Data.info()
Data.describe().transpose()

#splitting the dataset to Train and Test ,using 80% of data for training and 20% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

#standardization scaler - fit&transform on train and test
from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float))
X_test = s_scaler.fit_transform(X_test.astype(np.float))
print(X_train[1,:])
print(X_test[1,:])

# Creating a Neural Network Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
#loss functiion 
mse = MeanSquaredError()
#early stop callback if our model is unable to learn anymore
e_s = EarlyStopping(monitor='val_loss', patience=30)

#CNN model initiazation,using 6 layers and 1 output layer for regression result
model = Sequential()
model.add(Dense(320,activation='relu',input_shape=[len(X[1])]))
model.add(Dense(384,activation='relu'))
model.add(Dense(352,activation='relu'))
model.add(Dense(448,activation='relu'))
model.add(Dense(160,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))
model.compile(optimizer='Adam',loss='mse')
#100 epochs are enough after several trials
history = model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=64,epochs=100,verbose = 0,callbacks=[e_s])
#overview of model
model.summary()

loss_df = pd.DataFrame(model.history.history)
#ploting loss to overview training procedure
loss_df.plot(figsize=(12,8))
# Performing predictions
y_pred = model.predict(X_test)

y_test = y_test.reshape(-1,1)
#creating and printing dataframe to compare predicted values and actual values from the test set
dataf = pd.DataFrame({'y_pred': y_pred.ravel(), 'y_test': y_test.ravel()})
print(dataf)
#printing metrics like Mean Absolute Error to understand how efficient the model is
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))  
print('MSE:', metrics.mean_squared_error(y_test, y_pred))  
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('VarScore:',metrics.explained_variance_score(y_test,y_pred))

# Visualizing Our predictions
fig = plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test,'r')
plt.show()
#visualising the average values of error
error = y_pred - y_test
plt.hist(error, bins=20)
plt.xlabel('Prediction Error')
plt.show()

        
