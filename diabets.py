import numpy as np
import pandas as pd

#data
df = pd.read_csv("/Users/mac/Desktop/Python/src/datascience/diabets/diabetes.csv")
df.head()

df.info()
df.isnull().sum()
df.eq(0).sum()


df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']].replace(0,np.NaN)
df.fillna(df.mean(),inplace=True)
df.eq(0).sum()
df.corr()
import seaborn as sns
sns.heatmap(df.corr(), annot=True)

df.corr().nlargest(4,'Outcome').index

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
X = df[['Glucose','BMI','Age']]
y=df.iloc[:,8]

log_reg = linear_model.LogisticRegression()
log_reg_score = cross_val_score(log_reg,X,y,cv=10,scoring='accuracy').mean()
log_reg_score

result=[]
result.append(log_reg_score)

from sklearn import svm

linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm,X,y,cv=10,scoring='accuracy').mean()
linear_svm_score

import pickle

filename='diabets.sav'

log_reg.fit(X,y)

pickle.dump(log_reg,open(filename,'wb'))


loaded_model = pickle.load(open(filename,'rb'))

Glucose = 55
BMI = 60
Age = 20
prediction = loaded_model.predict([[Glucose,BMI,Age]])
prediction
