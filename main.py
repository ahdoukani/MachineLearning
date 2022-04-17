# This is a sample Python

import tensorflow as tensor
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle

from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

best_score = 0

for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
    # print(data.head())
    """linear model"""
    linear = linear_model.LinearRegression() # set general linear  regression model = variable called linear
    linear.fit(x_train,y_train)  # fit general lin regression model to fit x and y data in train sample
    acc = linear.score(x_test, y_test) #  R^2 value for the the fitted linear regression model
    print(acc) # acc=linear.score = R^2 = ((Variation of data around sample mean)- (Variation around model line)),
                                                            # /  variation of data around sample mean
    # variation or 'variance' is the sum  (deviation ^2)/n
    # variance = (std deviation)^2
    # r^2 tells us the proportion of the variation in the sample that is explained by our model
    # if you use the use the least squared error thn R= R^2
    # if you draw a random line the  to fit the data the R WIL NOT EQUAL R^2


    if acc > best_score:
        best_score = acc

        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print(best_score)
print("coefficient: \n", linear.coef_)
print("intercept: \n", linear.intercept_)
predictions = linear.predict(x_test)

for i in range(len(predictions)):
    # prints( predicted g3 (dependent), independent vars in test sample, actual g3 from data set
    print(predictions[i], x_test[i], y_test[i])

p = "G2"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("G3")
pyplot.show()
