#import pandas library
import pandas as panda

#open train.csv file
data= panda.read_csv("train.csv")

#use fillna to fill the null fields from Age and determine its Median
data["Age"] = data["Age"].fillna(data["Age"].median())
data.loc[data["Sex"] == "male", "Sex"] = 0
data.loc[data["Sex"] == "female", "Sex"] = 1
data["Embarked"] = data["Embarked"].fillna("S")
data.loc[data["Embarked"] == "S", "Embarked"] = 0
data.loc[data["Embarked"] == "C", "Embarked"] = 1
data.loc[data["Embarked"] == "Q", "Embarked"] = 2

#using sklearn functions, found snippets from an online source and incorporated in the model
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

alg = LinearRegression()

kf = KFold(data.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data["Survived"].iloc[train]
    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(data[predictors].iloc[test,:])
    predictions.append(test_predictions)

import numpy as np
predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
accuracy = sum(predictions[predictions == data["Survived"]]) / len(predictions)
