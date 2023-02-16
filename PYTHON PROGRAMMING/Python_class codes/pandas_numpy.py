# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 09:02:45 2020

@author: abdul
"""
# Python Packages for Data Analysis
## Numpy ia a general purpose array processing package, in which
# you can perform mathematical compution
import numpy as np
# creating an array
arr = np.array([[1,2,3],
               [4,2,5]])
# dimension of the array
arr.ndim
# to get type of array
type(arr)
# to get to know shape of array
arr.shape

# to know nuber of elements in your array
arr.size

# arange :- returns evenly spaced values within a given
# interval (step size specified)

# linspace: - returns evenly spcaed values

# reshaping array:- we can change the shape of array

# flatten array:- it helps to convert your mutlidimension to
# one dimension, it does has options called as row wise(default denoted by c).
# or column wise the specify as F


# create an array

a = np.array([[2,3,4],[5,8,7],[8,9,10]],dtype = 'float')

a.dtype

# creating an array from tuple
b = np.array((1,3,5))
b.size

# create a 3*4 array with all zeros inside

c = np.zeros((3,4))
c

# create an array with random integers

d = np.random.random((2,2))
d

# create a sequence of integers with specific values

f = np.arange(0,50,5)
f

# create a seqeunce of 10 values in range 0 to 5

g = np.linspace(0,5,10)
g

# reshapeing a 3*4 array to 2*2*3 array

ary = np.array([[2,3,4,5],[6,5,4,3],[4,5,8,9]])
ary

newary = ary.reshape(3,2,2,order = 'c')
newary.shape
newary

# flattern array

ar1 = np.array([[1,2,3],[4,5,6]])
ar1

ar2 = ar1.flatten()
ar2

# A list of elements in variable 'x'

x = [10,21,3,14,15,16]

# how to multiply the list values with 2
x*2 #  provides dual list

# Numpy array will help access the values
y = np.array(x)
type(y)
y*2
y>10
y[y>10]

# Numpy matrics
#Ex:1
a = np.matrix('1 2; 3 4')
a

#Ex:2
b = np.matrix([[1, 2], [3, 4]])
b


# Importing necessary libraries
import pandas as pd # importing pandas = > useful for creating dataframes
import numpy as np   # importing numpy = > useful for creating numpy arrays

# pandas we have series and data frame
# series is your single dimension
# dataframe is two dimension


x1 = [1, 2, 3, 4,5] # list format
x2 = [10, 11, 12,100,'NA']  # list format

x3 = list(range(5))

new_x = list(zip(x1,x2,x3))

# Creating a data frame using explicits lists
X = pd.DataFrame(new_x, columns= ['x1','x2','x3'],index = [101,102,103,104,105])
X
X.isnull()



# Importing necessary libraries
import pandas as pd # importing pandas = > useful for creating dataframes
import numpy as np   # importing numpy = > useful for creating numpy arrays

# pandas we have series and data frame
# series is your single dimension
# dataframe is two dimension


x1 = [1, 2, 3, 4,5,np.nan] # list format
x2 = [np.nan, 11, 12,100,np.nan,200]  # list format

x3 = list(range(6))

new_x = list(zip(x1,x2,x3))

# Creating a data frame using explicits lists
X = pd.DataFrame(new_x, columns= ['x1','x2','x3'],index = [0,1,2,3,4,5])
X.head()
# to get missing values
X.isnull()
X.columns
X.dtypes
#############################################
y= pd.Series(x1)  # Converting list format into pandas series format
z = pd.Series(x2) # Converting list format into pandas series format
w = pd.Series(x3)

c =y.append(z,ignore_index = True)
y
t = c.append(w,ignore_index= True)
t
# we can use .to_frame to get data frame out of series
t_1 = t.to_frame(name = 'new')
# concatenantion adding your column to the original data frame which is X
t_new = pd.concat([X,t_1],axis = 1).reindex(t_1.index)

t_new.head()
t_new.columns

# column names
#.rename function to change your column names
# df = df.rename(columns = {"key": "value", "key1": "value1",})
# key = old column name  and value is new column name
t_new = t_new.rename(columns = {"x1":"age"})

t_new = t_new.rename(columns = {"x2": "years","x3":"month"})


# accessing columns using "." (dot) operation
# dataframe.column name
X.x1
# accessing columns alternative way
X["x1"]

# Accessing multiple columns : giving column names as input in list format
X[["x1","x2"]]

# Accessing elements using ".iloc" : accessing each cell by row and column
# index values
# df.iloc
X.iloc[0:3,1]

X.iloc[:,:] # to get entire data frame

# Merge operation using pandas

df1 = pd.DataFrame({"A1":[1,2,3],"A2":[4,8,12]})
df2 = pd.DataFrame({"A1":[1,2,3,4],"A3":[4,8,12,15],})

merge = pd.merge(df1,df2, on = "A1") # merge function
merge
#joining operation
df1 = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],},index = {2000,2001,2002})
df2 = pd.DataFrame({"X3":[1,2,3],"X4":[4,8,12],},index = {2001,2002,2003})

joined = df2.join(df1) #join function based on index values
joined

# Replace index name
df = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12]})


df.set_index("X1", inplace = True) #Assiging index names using column names

# Change the column names
df = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],})

df  = df.rename(columns = {"X1":"X3"}) #Change column names

print(df)

# Concatenation

df1 = pd.DataFrame({"X1":[1,2,3],"X2":[4,8,12],},index = {2000,2001,2002})
df2 = pd.DataFrame({"X1":[4,5,6],"X2":[14,16,18],},index = {2003,2004,2005})

Concatenate = pd.concat([df1,df2])

print(Concatenate)

import pandas as pd
help(pd.read_csv) # to get csv files
help(pd.read_excel) # to get excel files
# Import data (.csv file) using pandas. We are using mba data set
mba = pd.read_csv("C:/Users/abdul/Desktop/assignmnent@360 key/mba.csv")
#mba1 = pd.read_csv("C:/Users/abdul/Desktop/assignmnent@360 key/mba.csv")
type(mba) # pandas data frame


#Matplotlib package
import matplotlib.pyplot as plt # mostly used for visualization purposes
import seaborn as sns # advance visualization
# We have considered MBA dataset and executed few important vizualizations
mba.head()
# information about your data frame
mba.describe()
# to get shape of dataframe
mba.shape
# to get information about columns
mba.columns
#Histogram
plt.hist(mba.gmat)
plt.hist(mba['gmat'])
plt.hist(mba['workex'])
help(plt.hist)
#Boxplot
plt.boxplot(mba['gmat']);plt.ylabel("GMAT")   # for vertical
plt.boxplot(mba['gmat'],1,'rs',0)# For Horizontal
help(plt.boxplot)
plt.boxplot(mba)

# scatter plot

plt.scatter(mba.workex,mba.gmat); plt.xlabel('workex'); plt.ylabel("mba")

# correlation coefficients

np.corrcoef(mba.workex,mba.gmat)
#Barplot
plt.bar
plt.bar(height = mba["gmat"], x = np.arange(773)) # initializing the parameter

#Import the dataset of cars
mtcars = pd.read_csv("C:/Users/abdul/Desktop/assignmnent@360 key/mtcars.csv")
mtcars.shape
mtcars.describe()
mtcars.gear.value_counts()
#Line Chart
mtcars.mpg.groupby(mtcars.gear).plot(kind="line")

# Pie chart
mtcars.cyl.value_counts().plot(kind="pie")

# Area Chart
mtcars.mpg.plot(kind='area')

#Scatter Plot
N = 32
colors = np.random.rand(N)
plt.scatter(mtcars.mpg, mtcars.wt,c= colors,alpha = 1,s = 20.6 )
plt.scatter(mtcars.mpg,mtcars.wt)

# correlation coefficients

np.corrcoef(mtcars.mpg,mtcars.wt)