#Raw material Collect → Clean → Convert → Organise → Split → Scale → Train → Predict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)
#X is all that we have given to the model
#Y is what we want the model to predict

#A machine learning model needs 2 things: -Information that we give to the model
#                                         -Answers that we want the model to predict

from sklearn.impute import SimpleImputer #Imputer means something that fills in the missing values
imputer = SimpleImputer(missing_values= np.nan,strategy="mean")

imputer.fit(X[:, 1:3]) #fit will look at the data and learn what is needed
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transform will do what is required

#Encoding the independant variables

#Categorical data -> Numerical data is called Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[
    ("encoder",OneHotEncoder(),[0])
],remainder="passthrough")
#ColumnTransformer: We want to apply a particular transformation to particular columns.
#Encoder: Just a name we give to this transformation.
#OneHotEncoder: Actual transformation we want to perform.
#remainder="passthrough" is for no transformation to be done to other columns

X = np.array(ct.fit_transform(X))

#Encoding the dependant variables

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
Y = le.fit_transform(Y)
#Learn the categories in Y (yes,no) and convert them into numerical data (yes=1,no=0)

#Split the dataset is import when we train and test the model on the same data, we don't get meaningful idea of how well the model will work on unseen data
#It's import to divide the dataset into training data (to teach the model) and testing data (used to check how well the model is performing)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state = 1)
#X_train = input data for training
#X_test = input data for testing
#Y_train = correct answers corresponding to the training data
#Y_test = correct answers corresponding to the testing data

#Test size means 20% of the data goes as testing set
#Random state controls the randomness used while splitting (0 or 1)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
