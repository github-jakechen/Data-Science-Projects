# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 02:20:33 2020

@author: dimen
"""


import datetime
print(datetime.datetime.now())

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

import itertools
import scipy

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))

###-------------------------------------------------------
###         Import and inspect data
###-------------------------------------------------------


import os
os.chdir("C:\\Users\\dimen\\Documents\\MMA\\MMA 869 Machine Learning & AI\\assignment\\")

#Read data
df= pd.read_csv("OJ.csv")

#Inspect data
list(df)
df.shape
df.info()
df.describe().transpose()
df.head(n=20)
df.tail()

###-------------------------------------------------------
###         Data preparation
###-------------------------------------------------------

#Set MM as 1 and CH as 0
df['Purchase'].replace('CH', 0,inplace=True)
df['Purchase'].replace('MM', 1,inplace=True)

#Recode dummy variables
df['Store7'].replace('No', 0,inplace=True)
df['Store7'].replace('Yes', 1,inplace=True)

#Rename unamed column
df.rename( columns={'Unnamed: 0':'Index'}, inplace=True )

#Inspect data
df.info()

#convert to array
X = df[['Index', 
       'WeekofPurchase',
       'StoreID',
       "PriceCH",
       "PriceMM",
       "DiscCH",
       "DiscMM",
       "SpecialCH",
       "SpecialMM",
       "LoyalCH",
       "SalePriceMM",
       "SalePriceCH",
       "PriceDiff",
       "Store7",
       "PctDiscMM",
       "PctDiscCH",
       "ListPriceDiff",
       "STORE"]].values
y = df['Purchase'].values

#Check that they are array
X[1:10,:]
y[1:10]

###-------------------------------------------------------
###         Helper functions
###-------------------------------------------------------
def plot_boundaries(X_train, X_test, y_train, y_test, clf, clf_name, ax, hide_ticks=True):
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02));
    
    
    score = clf.score(X_test, y_test);

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]);
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1];

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8);

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=100, cmap=cm_bright, edgecolors='k');
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap=cm_bright, edgecolors='k', alpha=0.6);

    ax.set_xlim(xx.min(), xx.max());
    ax.set_ylim(yy.min(), yy.max());
    if hide_ticks:
        ax.set_xticks(());
        ax.set_yticks(());
    else:
        ax.tick_params(axis='both', which='major', labelsize=18)
        #ax.yticks(fontsize=18);
        
    ax.set_title(clf_name, fontsize=28);
    ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=35, horizontalalignment='right');
    ax.grid();
    
    
    
def plot_roc(clf, X_test, y_test, name, ax, show_thresholds=True):
    y_pred_rf = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, y_pred_rf)

    ax.plot([0, 1], [0, 1], 'k--');
    ax.plot(fpr, tpr, label='{}, AUC={:.2f}'.format(name, auc(fpr, tpr)));
    ax.scatter(fpr, tpr);

    if show_thresholds:
        for i, th in enumerate(thr):
            ax.text(x=fpr[i], y=tpr[i], s="{:.2f}".format(th), fontsize=14, 
                     horizontalalignment='left', verticalalignment='top', color='black',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1', alpha=0.1));
        
    ax.set_xlabel('False positive rate', fontsize=18);
    ax.set_ylabel('True positive rate', fontsize=18);
    ax.tick_params(axis='both', which='major', labelsize=18);
    ax.grid(True);
    ax.set_title('ROC Curve', fontsize=18)


###-------------------------------------------------------
###         Data split to train/test
###-------------------------------------------------------


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



###-------------------------------------------------------
###         Decision Tree
###-------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=42, criterion="entropy",
                             min_samples_split=10, min_samples_leaf=10, max_depth=3, max_leaf_nodes=5)
clf.fit(X_train, y_train)

y_pred_dt = clf.predict(X_test)

class_names = [str(x) for x in clf.classes_]

#Model parameters
print(clf.tree_.node_count)
print(clf.tree_.impurity)
print(clf.tree_.children_left)
print(clf.tree_.threshold)



###-------------------------------------------------------
###         Decision Tree - Confusion Matrix
###-------------------------------------------------------

import seaborn as sn

confusion_data_dt = {'y_Actual': y_test,
        'y_Predicted': y_pred_dt
        }

df_dt = pd.DataFrame(confusion_data_dt, columns=['y_Actual','y_Predicted'])
confusion_matrix_dt = pd.crosstab(df_dt['y_Actual'], df_dt['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix_dt, annot=True)
plt.show()

###-------------------------------------------------------
###         Decision Tree - Classification report
###-------------------------------------------------------

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_dt, target_names=class_names))


#
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, log_loss

print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_dt)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_dt)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_dt)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_dt)))


#Classification Report
from yellowbrick.classifier import ClassificationReport

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(clf, classes=class_names, support=True)

visualizer.fit(X_train, y_train)  # Fit the visualizer and the model
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data


#Class Prediction Error
from yellowbrick.classifier import ClassPredictionError

visualizer = ClassPredictionError(clf, classes=class_names)

visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
g = visualizer.poof()

###-------------------------------------------------------
###         Decision Tree - ROC
###-------------------------------------------------------

    
from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(clf, classes=class_names)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data



###-------------------------------------------------------
###         Naive Bayes
###-------------------------------------------------------

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)
gnb

y_pred_gnb = gnb.predict(X_test)


###-------------------------------------------------------
###         Naive Bayes - Confusion Matrix
###-------------------------------------------------------


confusion_data_gnb = {'y_Actual': y_test,
        'y_Predicted': y_pred_gnb
        }

df_gnb = pd.DataFrame(confusion_data_gnb, columns=['y_Actual','y_Predicted'])
confusion_matrix_gnb = pd.crosstab(df_gnb['y_Actual'], df_gnb['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix_gnb, annot=True)
plt.show()

#Model parameter
gnb.theta_ # Mean of each feature per class
gnb.sigma_ # Variance of each feature per class


###-------------------------------------------------------
###         Naive Bayes - Classification report
###-------------------------------------------------------


#Classification report
print(classification_report(y_test, y_pred_gnb, target_names=class_names))

print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_gnb)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_gnb)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_gnb)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_gnb)))

###-------------------------------------------------------
###         Naive Bayes - ROC
###-------------------------------------------------------


from yellowbrick.classifier import ROCAUC

visualizer = ROCAUC(gnb, classes=class_names)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

###-------------------------------------------------------
###         KNN
###-------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)

y_pred_knn = knn_clf.predict(X_test)



#Model parameters
knn_clf.effective_metric_
knn_clf.effective_metric_params_

###-------------------------------------------------------
###         KNN - Confusion Matrix
###-------------------------------------------------------

confusion_data_knn = {'y_Actual': y_test,
        'y_Predicted': y_pred_knn
        }

df_knn = pd.DataFrame(confusion_data_knn, columns=['y_Actual','y_Predicted'])
confusion_matrix_knn = pd.crosstab(df_knn['y_Actual'], df_knn['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix_knn, annot=True)
plt.show()

###-------------------------------------------------------
###         KNN - Classification report
###-------------------------------------------------------



print(classification_report(y_test, y_pred_knn, target_names=class_names))

print("Accuracy = {:.2f}".format(accuracy_score(y_test, y_pred_knn)))
print("Kappa = {:.2f}".format(cohen_kappa_score(y_test, y_pred_knn)))
print("F1 Score = {:.2f}".format(f1_score(y_test, y_pred_knn)))
print("Log Loss = {:.2f}".format(log_loss(y_test, y_pred_knn)))

###-------------------------------------------------------
###         KNN - ROC
###-------------------------------------------------------

visualizer = ROCAUC(knn_clf, classes=class_names)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
g = visualizer.poof()             # Draw/show/poof the data

