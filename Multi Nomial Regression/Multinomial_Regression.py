### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mode = pd.read_csv("C:\\Users\\vaibh\\Desktop\\360 Digitmg\\Multi Nomial Regression\\mode.csv")
mode.head(10)

mode.describe()
mode.choice.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "choice", y = "cost.car", data = mode)
sns.boxplot(x = "choice", y = "cost.carpool", data = mode)
sns.boxplot(x = "choice", y = "cost.bus", data = mode)
sns.boxplot(x = "choice", y = "cost.rail", data = mode)
sns.boxplot(x = "choice", y = "time.car", data = mode)
sns.boxplot(x = "choice", y = "time.bus", data = mode)
sns.boxplot(x = "choice", y = "time.rail", data = mode)


# Scatter plot for each categorical choice of car
sns.stripplot(x = "choice", y = "cost.car", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.carpool", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.carpool", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "cost.rail", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.car", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.bus", jitter = True, data = mode)
sns.stripplot(x = "choice", y = "time.rail", jitter = True, data = mode)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "choice") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions

# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
