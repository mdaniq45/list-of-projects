import numpy as np
import pandas as pd
 
import matplotlib.pyplot as plt
df=pd.read_csv(r"C:\Users\hi\Downloads\Salary_Data.csv")
df
df.info()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn .linear_model import LinearRegression

regression=LinearRegression()
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)
compresion=pd.DataFrame()({'Actual':y_test, 'predict':y_pred})
plt.scatter(x_train, y_test, color ='red')
plt.plot(x_train,regression.predict(x_train),color="blue")
         
