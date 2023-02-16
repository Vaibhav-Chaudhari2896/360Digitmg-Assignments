import pandas as pd
import numpy as np

letters = pd.read_csv("C:\\Users\\vaibh\\Desktop\\360 Digitmg\\SVM\\letterdata.csv")
letters.describe()

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(letters, test_size = 0.20)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

