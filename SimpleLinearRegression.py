from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics  

LR= LinearRegression()
filepath="student_scores.csv"
data=pd.read_csv(filepath)

print(data.describe())

x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

#spliting data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#model fit
LR.fit(X_train,y_train)

##print(X_train)
##print(y_train)
##print(X_test)
##print(y_test)
##print(LR.intercept_)
##print(LR.coef_)

#y prediction
y_pred = LR.predict(X_test)  
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

##Y=b+mx
##Y=LR.intercept_+LR.coef_*(2.5)
##print(Y)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
