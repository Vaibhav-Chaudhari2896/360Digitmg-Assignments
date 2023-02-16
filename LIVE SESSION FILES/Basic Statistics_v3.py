2 + 2 # Function F9
# works as calculator

# Python Libraries (Packages)
# pip install <package name> - To install library (package), execute the code in Command prompt
# pip install pandas

import pandas as pd

dir(pd)

# Read data into Python
education = pd.read_csv(r"C:\Data\education.csv")
Education = pd.read_csv("C:/Data/education.csv")

A=10
a=10.1

education.info()

# C:\Users\education.csv - this is windows default file path with a '\'
# C:\\Users\\education.csv - change it to '\\' to make it work in Python

# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
education.workex.mean() # '.' is used to refer to the variables within object
education.workex.median()
education.workex.mode()

# pip install numpy
from scipy import stats
stats.mode(education.workex)

# Measures of Dispersion / Second moment business decision
education.workex.var() # variance
education.workex.std() # standard deviation
range = max(education.workex) - min(education.workex) # range
range

# Third moment business decision
education.workex.skew()
education.gmat.skew()

# Fourth moment business decision
education.workex.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

education.shape

# barplot
plt.bar(height = education.gmat, x = np.arange(1, 774, 1)) # initializing the parameter

plt.hist(education.gmat) # histogram
plt.hist(education.workex, color='red')

help(plt.hist)

plt.figure()

plt.boxplot(education.gmat) # boxplot

help(plt.boxplot)

###############################################
############# Data Preprocessing ##############

################ Type casting #################
import pandas as pd

data = pd.read_csv("C:/Data/ethnic diversity.csv")
data.dtypes

help(data.astype)
# Now we will convert 'float64' into 'int64' type. 
data.Salaries = data.Salaries.astype('int64')
data.dtypes

data.age = data.age.astype('float32')
data.dtypes

###############################################
### Identify duplicates records in the data ###
data = pd.read_csv("C:/Data/mtcars_dup.csv")

duplicate = data.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates
data1 = data.drop_duplicates()

################################################
############## Outlier Treatment ###############
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Data/ethnic diversity.csv")
df.dtypes

# let's find outliers in Salaries
sns.boxplot(df.Salaries)

sns.boxplot(df.age)
# No outliers in age column

# Detection of outliers (find limits for salary based on IQR)
IQR = df['Salaries'].quantile(0.75) - df['Salaries'].quantile(0.25)
lower_limit = df['Salaries'].quantile(0.25) - (IQR * 1.5)
upper_limit = df['Salaries'].quantile(0.75) + (IQR * 1.5)

############### 1. Remove (let's trim the dataset) ################
# Trimming Technique
# let's flag the outliers in the data set
outliers_df = np.where(df['Salaries'] > upper_limit, True, np.where(df['Salaries'] < lower_limit, True, False))
sum(outliers_df)

df_trimmed = df.loc[~(outliers_df), ]
df.shape, df_trimmed.shape

# let's explore outliers in the trimmed dataset
sns.boxplot(df_trimmed.Salaries)
# we see no outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit
df['df_replaced'] = pd.DataFrame(np.where(df['Salaries'] > upper_limit, upper_limit, np.where(df['Salaries'] < lower_limit, lower_limit, df['Salaries'])))
sns.boxplot(df.df_replaced)

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
# conda install -c conda-forge feature_engine
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Salaries'])

df_t = winsor.fit_transform(df[['Salaries']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Salaries)

################################################
#### zero variance or near zero variance ######

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

df.var() # variance of numeric variables
df.var() == 0

############################################################################
###########
# Discretization

import pandas as pd
data = pd.read_csv("C:/Data/ethnic diversity.csv")
data.head()

data.info()
data.describe()
data['Salaries_new'] = pd.cut(data['Salaries'], bins=[min(data.Salaries), 
                                                  data.Salaries.mean(), max(data.Salaries)], labels=["Low","High"], include_lowest=True)
data.head(10)

data.Salaries_new.value_counts()
data.MaritalDesc.value_counts()

############################################################################
#################### Missing Values - Imputation ###########################
import numpy as np
import pandas as pd

# load the dataset
# use modified ethnic dataset
df = pd.read_csv('C:/Data/modified ethnic.csv') # for doing modifications

# check for count of NA's in each column
df.isna().sum()

# Create an imputer object that fills 'Nan' values
# Mean and Median imputer are used for numeric data (Salaries)
# Mode is used for discrete data (ex: Position, Sex, MaritalDesc)

# for Mean, Meadian, Mode imputation we can use Simple Imputer or df.fillna()
from sklearn.impute import SimpleImputer

# Mean Imputer 
mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
df["Salaries"] = pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
df["Salaries"].isna().sum()

# Median Imputer
median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df["age"] = pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
df["age"].isna().sum()  # all records replaced by median 

df.isna().sum()

# Mode Imputer
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

df["Sex"] = pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))

df["MaritalDesc"] = pd.DataFrame(mode_imputer.fit_transform(df[["MaritalDesc"]]))
df.isnull().sum()  # all Sex, MaritalDesc records replaced by mode

##################################################
################## Dummy Variables ###############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# we use ethinc diversity dataset
df = pd.read_csv("C:/Data/ethnic diversity.csv")

df.columns # column names
df.shape # will give u shape of the dataframe

# drop emp_name column
df.drop(['Employee_Name','EmpID','Zip'], axis=1, inplace=True)
df.dtypes

# Create dummy variables
df_new = pd.get_dummies(df)
df_new_1 = pd.get_dummies(df, drop_first = True)
# we have created dummies for all categorical columns

##### One Hot Encoding works
df.columns
df = df[['Salaries', 'age', 'Position', 'State','Sex',
         'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department','Race']]

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method

enc_df = pd.DataFrame(enc.fit_transform(df.iloc[:, 2:]).toarray())


#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# Creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = df.iloc[:, 0:9]

y = df['Race']
y = df.iloc[:, 9:] # Alternative approach

df.columns

X['Sex'] = labelencoder.fit_transform(X['Sex'])
X['MaritalDesc'] = labelencoder.fit_transform(X['MaritalDesc'])
X['CitizenDesc'] = labelencoder.fit_transform(X['CitizenDesc'])

### label encode y ###
y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

### We have to convert y to data frame so that we can use concatenate function
# concatenate X and y
df_new = pd.concat([X, y], axis = 1)

## rename column name
df_new.columns
df_new = df_new.rename(columns={0:'Race'})


#####################
# Normal Quantile-Quantile Plot

import pandas as pd

# Read data into Python
education = pd.read_csv("C:/Data/education.csv")

import scipy.stats as stats
import pylab

# Checking Whether data is normally distributed
stats.probplot(education.gmat, dist="norm", plot=pylab)

stats.probplot(education.workex, dist="norm", plot=pylab)

import numpy as np

# Transformation to make workex variable normal
stats.probplot(np.log(education.workex), dist="norm", plot=pylab)



####################################################
######## Standardization and Normalization #########
import pandas as pd
import numpy as np

### Standardization
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("C:/Data/mtcars.csv")

a = data.describe()
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
df = scaler.fit_transform(data)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load data set
ethnic = pd.read_csv("C:/Data/ethnic diversity.csv")
ethnic.columns
ethnic.drop(['Employee_Name', 'EmpID', 'Zip'], axis = 1, inplace = True)

a1 = ethnic.describe()

# Get dummies
ethnic = pd.get_dummies(ethnic, drop_first = True)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i): 
    x = (i-i.min())/(i.max()-i.min())
    return(x)

df_norm = norm_func(ethnic)
b = df_norm.describe()


#####################

import scipy.stats as stats

# z-distribution
# cdf => cumulative distributive function
stats.norm.cdf(680, 711, 29)  # Given a value, find the probability

# ppf => Percent point function;
stats.norm.ppf(0.025, 0, 1) # Given probability, find the Z value

# t-distribution
stats.t.cdf(1.98, 139) # Given a value, find the probability
stats.t.ppf(0.025, 139) # Given probability, find the t value


################################################################
#################### String manipulation #######################
word = "Keep Adapting"
print(word)

word

# Accessing
word = "Keep Adapting"

letter = word[4]

print(letter)

# length 
word = "Keep Adapting"

len(word)

letters = "wenf bwehfwfnewfje    "
len(letters)

#finding
word = "Keep Adapting"
print(word.count('p')) # count how many times p is in the string
print(word.find("p")) # find the letter in the string
print(word.index("Adapting")) # find the letters Adapting in the string

s = "The world won't care about your self-esteem. The world will expect you to accomplish something BEFORE you feel good about yourself."

print(s.count(' '))

# Slicing
y = "             "
print(y.count(' '))

word1 = "_$_the internet frees us from the responsibility of having to retain anything in our long-term memory@_."

print (word1[0])
print(word1[-1]) #get one char of the word
print (word1[0:1]) #get one char of the word (same as above)
print (word1[0:3]) #get the first three char
print (word1[:3]) #get the first three char
print (word1[-3:]) #get the last three char
print (word1[3:]) #get all but the three first char
print (word1[:-3]) #get all but the three last character
print (word1[3:-3]) #get all 


# spliting
word3 = 'Good health is not something we can buy. However, it can be an extremely valuable savings account.'

a = word3.split(' ')
a # Split on whitespace

['Good','health','is','not','something','we','can','buy.','However,','it','can','be','an','extremely','valuable','savings','account.']
type(a)

# Startswith / Endswith
word3 = 'Remain calm, because peace equals power.'
word3.startswith("R")
word3.endswith("e")
word3.endswith(".")

# Repeat string 
print( " * "* 10 )# 

# Replacing
word4 = "Live HapLive"

word4.replace("Live", "Lead Life")

# dir(string)

# Reversing
string = "eert a tsniaga pu mih deit yehT .meht htiw noil eht koot dna tserof eht otni emac sretnuh wef a ,yad enO "

print (''.join(reversed(string)))

#Strip
#Python strings have the strip(), lstrip(), rstrip() methods for removing
#any character from both ends of a string.

# If the characters to be removed are not specified then white-space will be removed
