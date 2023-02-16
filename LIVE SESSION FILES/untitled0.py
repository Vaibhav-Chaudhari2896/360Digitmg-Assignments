import pandas as pd;
data=pd.read_csv('C:\\Users\\vaibh\\Desktop\\360 Digitmg\\education.csv')
data
data.info(verbose=True)
data.dtypes
data.gmat=data.gmat.astype('float32')


data.shape
data.head()
# Describe the data 
data.describe()

# Mean, Median, Mode for Workex
data.workex.mean()
data.workex.median()
data.gmat.mode()

# Mean, Median, Mode for GMAT
data.gmat.mean()
data.gmat.median()
data.gmat.mode()

# Variance
data.workex.var()
data.gmat.var()

#Standard Deviation
data.workex.std()
data.gmat.std()

# Range
data.gmat.max()-data.gmat.min()
data.workex.max()-data.workex.min()

#Skewness
data.gmat.skew()
plt.hist(data.gmat)
data.workex.skew()

#kurtosis
data.gmat.kurt()
data.workex.kurt()

import matplotlib.pyplot as plt
import numpy as np

# Histogram for data
plt.hist(data.gmat)
plt.hist(data.workex)

# Box Plot
plt.boxplot(data.gmat)
plt.boxplot(data.workex,vert=False)

#bar plot

plt.bar(np.arange(data.shape[0]), data.gmat)

data.duplicated().sum()
data.drop_duplicates()
