#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# import dataset 
zoo = pd.read_csv("zoo.csv")
print(zoo.head())
print(zoo.shape)
# check for null values
print(zoo.isnull().any())
# explore data
print(zoo.info())
desc = zoo.describe()

# split dataset into two dataframes
# X contains inputs
# y contains target
X = zoo.drop(['class_type','animal_name'], axis=1).values
y = zoo['class_type'].values

# split data into training set and testing set
# 80% training set and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # KNN classifier model for k=3
# knn3 = KNeighborsClassifier(n_neighbors=3)
# knn3.fit(X_train, y_train)
# print("KNN train score for k=3:")
# print(knn3.score(X_train, y_train))
# print("KNN test score for k=3:")
# print(knn3.score(X_test, y_test))

# # run prediction
# y_pred = knn3.predict(X_test)
# print("Confusion matrix:")
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))

# # plot visualisation
# plt.rcParams['figure.figsize'] = (9,9)
# _, ax = plt.subplots()
# ax.hist(y_test, color = 'b', alpha = 0.5, label = 'actual', bins=7)
# ax.hist(y_pred, color = 'g', alpha = 0.5, label = 'prediction', bins=7)
# ax.yaxis.set_ticks(np.arange(0,13))
# ax.legend(loc = 'best')
# plt.show()


# find optimal value of k
# get scores for k values from 1 to 50
k_list = np.arange(1,50,2)
mean_scores = []
accuracy_list = []
error_rate = []

for i in k_list:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    score = cross_val_score(knn,X_train, y_train,cv=10)
    mean_scores.append(np.mean(score))
    error_rate.append(np.mean(pred_i != y_test))
    
print("Mean scores:")
print(mean_scores)
print("Error rate:")
print(error_rate)

# plot average accuracy score for each k value
plt.figure()
plt.plot(k_list,mean_scores,marker='o')
plt.title('Accuracy scores for different K values')
plt.xlabel('K value')
plt.ylabel('Mean accuracy score')
plt.xticks(k_list)
plt.rcParams['figure.figsize']=(12,12)
plt.show()

# plot average error rate for each k value
plt.figure()
plt.plot(k_list,error_rate,marker='o')
plt.title('Error rate for different K values')
plt.xlabel('K value')
plt.ylabel('Error rate')
plt.xticks(k_list)
plt.rcParams['figure.figsize']=(12,12)
plt.show()
