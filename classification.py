import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from random import random
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR

data = load_breast_cancer(as_frame=True)
df = data.frame

print("Total diagnosis are : ",df.shape[0])
print("Malignant (0) : ",df.target.value_counts()[0])
print("Benign (1) : ",df.target.value_counts()[1])

featureMeans = list(df.columns[1:11])
x = df.loc[:,featureMeans]
y = df.loc[:,'target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=49)

def NaiveBayes(x_train,x_test,y_train,y_test):
    clf = GaussianNB().fit(x_train,y_train)
    prediction = clf.predict(x_test)

    print("\n\nNaive Bayes Model:")
    print("Accuracy of GaussianNB Classifier on training set : ",clf.score(x_train,y_train))
    print("Accuracy of GaussianNB Classifier on test set : ",clf.score(x_test,y_test))
    print("Classification report for classifier %s :\n%s" %(clf,metrics.classification_report(y_test,prediction)))
    print("Confusion Matrix : \n%s" % (metrics.confusion_matrix(y_test,prediction)))

def KNN(x_train,x_test,y_train,y_test):
    clf = KNeighborsClassifier().fit(x_train,y_train)
    prediction = clf.predict(x_test)

    print("\n\nKNN Model:")
    print("Accuracy of KNN Classifier on training set : ",clf.score(x_train,y_train))
    print("Accuracy of KNN Classifier on test set : ",clf.score(x_test,y_test))
    print("Classification report for classifier %s :\n%s" %(clf,metrics.classification_report(y_test,prediction)))
    print("Confusion Matrix : \n%s" % (metrics.confusion_matrix(y_test,prediction)))

def tuned_KNN(x_train,x_test,y_train,y_test):
    clf = KNeighborsClassifier(weights='distance',algorithm='ball_tree').fit(x_train,y_train)
    prediction = clf.predict(x_test)

    print("\n\nTuned KNN Model:")
    print("Accuracy of Tuned KNN Classifier on training set : ",clf.score(x_train,y_train))
    print("Accuracy of Tuned KNN Classifier on test set : ",clf.score(x_test,y_test))
    print("Classification report for classifier %s :\n%s" %(clf,metrics.classification_report(y_test,prediction)))
    print("Confusion Matrix : \n%s" % (metrics.confusion_matrix(y_test,prediction)))

def LogisticReg(x_train,x_test,y_train,y_test):
    lr = LR().fit(x_train,y_train)
    prediction = lr.predict(x_test)

    print("\n\nLogistic Regression Model:")
    print("Accuracy of Logistic Regression on training set : ",lr.score(x_train,y_train))
    print("Accuracy of Logistic Regression on test set : ",lr.score(x_test,y_test))
    print("Classification report for classifier %s :\n%s" %(lr,metrics.classification_report(y_test,prediction)))
    print("Confusion Matrix : \n%s" % (metrics.confusion_matrix(y_test,prediction)))

NaiveBayes(x_train,x_test,y_train,y_test)
KNN(x_train,x_test,y_train,y_test)
tuned_KNN(x_train,x_test,y_train,y_test)
LogisticReg(x_train,x_test,y_train,y_test)

plt.figure(figsize=(5,5))
sns.heatmap(df[featureMeans].corr(),annot=True,square=True)
plt.show()