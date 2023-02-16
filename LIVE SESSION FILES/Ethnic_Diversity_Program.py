import pandas as pd;
data1=pd.read_csv('C:\\Users\\vaibh\\Desktop\\360 Digitmg\\ethnic diversity.csv')
#data2 = open('C:\\Users\\vaibh\\Desktop\\360 Digitmg\\ethnic diversity.csv','r')
#data2.read(delimiter=',')
data1
data1.info()

# Describe the data 
data1.describe()

# Mean, Median, Mode for Workex
data1.age.mean()
data1.age.median()
data1.age.mode()

# Mean, Median, Mode for GMAT
data1.Salaries.mean()
data1.Salaries.median()
data1.Salaries.mode()

# Variance
data1.age.var()
data1.Salaries.var()

#Standard Deviation
data1.age.std()
data1.Salaries.std()

# Range
data1.age.max()-data1.age.min()
data1.Salaries.max()-data1.Salaries.min()

#Skewness
data1.age.skew()
data1.Salaries.skew()

#kurtosis
data1.age.kurt()
data1.Salaries.kurt()

import matplotlib.pyplot as plt
import numpy as np

# Histogram for data
plt.hist(data1.age)
plt.hist(data1.Salaries)

# Box Plot
plt.boxplot(data1.age)
plt.boxplot(data1.Salaries,vert=False)

#bar plot

plt.bar(np.arange(data1.shape[0]), data1.age)
plt.bar(np.arange(data1.shape[0]), data1.Salaries)

type(data1.age)
data1.dtypes

data1.Salaries=data1.Salaries.astype('int64')
data1.dtypes

duplicate=data1.duplicated()
duplicate
data1


import seaborn as sns
sns.boxplot(data1.Salaries)

IQR=data1.Salaries.quantile(0.75)-data1.Salaries.quantile(0.25)
IQR
lower_limit=data1.Salaries.quantile(0.25)-(1.5*IQR)
upper_limit=data1.Salaries.quantile(0.75)+(1.5*IQR)

# As salaries cannot be lower than 0
lower_limit=0

outliers=np.where((data1.Salaries > upper_limit) | (data1.Salaries < lower_limit),True,False)
outliers.sum()

df_trimmed=data1.loc[~(outliers), ]
df_trimmed.shape
data1.shape

df_trimmed_new=data1.drop(data1[outliers == True].index)
df_trimmed_new.shape

# Replacing the values
data2=data1.copy()
data2['Salaries_updated'] = pd.DataFrame(np.where(data2.Salaries > upper_limit,upper_limit, np.where(data2.Salaries < lower_limit,lower_limit,data2.Salaries)))
data2

from feature_engine.outliers import Winsorizer
np.loadtxt('C:\\Users\\vaibh\\Desktop\\360 Digitmg\\ethnic diversity.csv',)






data1.head()
data1.describe()
data1['Salaries_new']=pd.cut(data1['Salaries'],bins=[min(data1.Salaries),data1.Salaries.mean(),max(data1.Salaries)],labels=['Low','High'],include_lowest=True)
data1

data1.Salaries_new.value_counts()
data1.isna().sum()

data1
data1.drop(['Employee_Name','EmpID','Zip'],axis=1,inplace=True)
data1.dtypes

df_new=pd.get_dummies(data1,drop_first=True)
df_new.shape

from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
enc_df=pd.DataFrame(enc.fit_transform(data1.iloc[:,2:]).toarray())
enc_df

y=pd.DataFrame(data1['Race'])
x=data1.iloc[:,0:9]
z=pd.concat(x,y)

fill_value
