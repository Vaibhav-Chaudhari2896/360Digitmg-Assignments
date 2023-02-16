import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/vaibh/Desktop/360 Digitmg/Decision Tree/credit.csv")

data.isnull().sum()
data.dropna()

data.columns
data.info()

data = data.drop(["phone"], axis = 1)

desc = data.describe()

# Converting into Numeric
lb = LabelEncoder()
data["checking_balance"] = lb.fit_transform(data["checking_balance"])
data["credit_history"] = lb.fit_transform(data["credit_history"])
data["purpose"] = lb.fit_transform(data["purpose"])
data["savings_balance"] = lb.fit_transform(data["savings_balance"])
data["employment_duration"] = lb.fit_transform(data["employment_duration"])
data["other_credit"] = lb.fit_transform(data["other_credit"])
data["housing"] = lb.fit_transform(data["housing"])
data["job"] = lb.fit_transform(data["job"])

#data["default"]=lb.fit_transform(data["default"])

data['default'].unique()
data['default'].value_counts()

colnames = list(data.columns)

predictors = colnames[:15]
target = colnames[15]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy


# Automatic Tuning - Hyperparameters
######
# GridSearchCV

from sklearn.model_selection import GridSearchCV

model = DT(criterion = 'entropy')

param_grid = {'min_samples_leaf': [1, 5, 10, 20],
              'max_depth': [2, 4, 6, 8, 10],
              'max_features': ['sqrt']}


grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                                scoring = 'accuracy', n_jobs = -1, cv = 5, 
                                refit=True, return_train_score=True)

grid_search.fit(train[predictors], train[target])


grid_search.best_params_

cv_dt_clf_grid = grid_search.best_estimator_

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(test[target], cv_dt_clf_grid.predict(test[predictors]))
accuracy_score(test[target], cv_dt_clf_grid.predict(test[predictors]))

# Evaluation on Training Data
confusion_matrix(train[target], cv_dt_clf_grid.predict(train[predictors]))
accuracy_score(train[target], cv_dt_clf_grid.predict(train[predictors]))


######
# RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV

model = DT(criterion = 'entropy')

param_dist = {'min_samples_leaf': list(range(1, 50)),
              'max_depth': list(range(2, 20)),
              'max_features': ['sqrt']}

n_iter = 50

model_random_search = RandomizedSearchCV(estimator = model,
                                         param_distributions = param_dist,
                                         n_iter = n_iter)

model_random_search.fit(train[predictors], train[target])

model_random_search.best_params_

dT_random = model_random_search.best_estimator_

#prediciton on test data 
pred_random = dT_random.predict(test[predictors])
pd.crosstab(test[target], pred_random, rownames=['Actual'], colnames=['Predictions'])

np.mean(pred_random == test[target])

#predicition on train data 
pred_random = dT_random.predict(train[predictors])
pd.crosstab(train[target], pred_random, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(pred_random == train[target])
