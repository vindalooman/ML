# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:27:31 2018
Based on https://machinelearningmastery.com tutorial
@author: vinda
"""

# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

#Dimensions of Dataset

#We can get a quick idea of how many instances (rows) 
#and how many attributes (columns) the data contains with the shape property.
# shape

print(dataset.shape)

#let's have a look to the data
# head
print(dataset.head(20))

#let's look to dataset stats
# descriptions
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())


#Univariate plots to better understand each attribute.
#Multivariate plots to better understand the relationships between attributes.

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

#interactions between the variables.

#scatterplots of all pairs of attributes. 
#This can be helpful to spot structured relationships between input variables.
# scatter plot matrix
scatter_matrix(dataset)
plt.show()

#evalutating algo


#Separate out a validation dataset.
#Set-up the test harness to use 10-fold cross validation.
#Build 5 different models to predict species from flower measurements
#Select the best model.



# Split-out validation dataset
# X contains the parameters
# Y contains the result
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
#sklearn model_selection.train_test_split to split dataset into training and validation subdatasets
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#Test Harness
#10-fold cross validation .
#split test dataset into 10 parts, train on 9 and test on 1 
#repeat for all combinations of train-test splits.

#metric of ‘accuracy‘ to evaluate models

# Test options and evaluation metric
#  a ratio of the number of correctly predicted instances 
#divided by the total number of instances in the dataset 
#multiplied by 100 to give a percentage (e.g. 95% accurate)
seed = 7
scoring = 'accuracy'

# idea from the plots that some of the classes are partially linearly separable in some dimensions

#evaluate 6 different algorithms:

#Logistic Regression (LR)
#Linear Discriminant Analysis (LDA)
#K-Nearest Neighbors (KNN).
#Classification and Regression Trees (CART).
#Gaussian Naive Bayes (NB).
#Support Vector Machines (SVM).

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()    

# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))