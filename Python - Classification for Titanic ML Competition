#%% Kaggle competition - Titanic: Machine Learning from Disaster

from datetime import date
import sys

import inline as inline
import pandas

today = date.today()
print("As of:", today)
print("Author: ", "Jake Chen")
print("Python version:", sys.version)

#%%         Import libraries
#---------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%         Import data
#---------------------------------------------------------------

import os
os.chdir("C:\\Users\\dimen\\Documents\\Python\\Kaggle - Titanic\\")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


TrainTest = 'Set'
df_train[TrainTest] = 'Train'
df_test[TrainTest] = 'Test'
df = df_train.append(df_test) #Append datasets

#%%         Inspect data
#---------------------------------------------------------------

df.info()
df.describe()


#%%         Handling missing values
#---------------------------------------------------------------

df.isnull().sum() #count of missing values in each column
df.isnull().sum() /len(df)*100 #percentage of missing values in each column

df.isnull().any().sum() #Number of columns with missing values
df.isnull().any(axis=1).sum() #Number of rows with missing values

#Quick & dirty: Remove missing rows by specific columns
#df.dropna(subset=['Age', 'Fare', 'Embarked', 'Embarked_num'], inplace=True)

#Check distribution
plt.hist(df['Age'], bins=20)
plt.title('Age histogram')
plt.show()

plt.boxplot(df['Age'][~np.isnan(df['Age'])])
plt.title('Age boxplot')
plt.show() #Outliers exist
#Boxplot is modified to not look at missing values (Boxplot doesn't work with missing values)

plt.hist(df['Fare'], bins=20)
plt.title('Fare histogram')
plt.show()


#Remove outliers (IQR score)
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)
IQR = Q3 - Q1
print(IQR)
lower_range = Q1 - (1.5 * IQR)
upper_range = Q3 + (1.5 * IQR)

#Replace Age < 1 with NaN
df['Age'].mask(df['Age'] < 1, np.nan, inplace=True)


#Impute missing values for features: Age, Fare, and Embarked (ignore Cabin)

    #Mean imputation
df['Age'].fillna(df['Age'].mean(), inplace=True) #Replace NaN with mean permanently
df['Fare'].fillna(df['Fare'].mean(), inplace=True) #Replace NaN with mean permanently
df['Embarked'].fillna(df['Embarked'].mode(), inplace=True) #Replace NaN with mode permanently


#%%         Handling categorical values
#---------------------------------------------------------------

#Binary encoding
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})

#Numeric encoding
from sklearn.preprocessing import LabelEncoder
#df['Ticket_num'] = LabelEncoder().fit_transform(df['Ticket'])
#Don't use LabelEncoder with missing values, it will encode the NaN as a non-NaN category

#Note: there is a lot of extractable information in 'Ticket'


#Onehot encoding
#Onehot encoder only takes numerical categorical values, hence any value of string type should be label encoded before one-hot encoded

#Pandas also offer a way to onehot encode
df = pd.get_dummies(df, columns=['Embarked'], prefix=['Embarked'])

#%%         Scaling data
#---------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']
df[num_features] = scaler.fit_transform(df[num_features])

#%%         Check data imbalance
#---------------------------------------------------------------

df['Survived'].value_counts()
class_count_0, class_count_1 = df['Survived'].value_counts()
imbalance_ratio = class_count_1/class_count_0
print('imbalance ratio: ', imbalance_ratio)


#%%         Train test split
#---------------------------------------------------------------

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
label = 'Survived'
X, y = df[df[TrainTest] == 'Train'][features].values, df[df[TrainTest] == 'Train'][label].values

from sklearn.model_selection import train_test_split

# Split data 70%-30% into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=123)


#%%         Logistic regression
#---------------------------------------------------------------

from sklearn.linear_model import LogisticRegression

# Set regularization rate: used to counteract any bias in the sample, and help the model generalize well by avoiding overfitting the model to the training data
reg = 0.01

# train a logistic regression model on the training set
model = LogisticRegression(C=1/reg, solver="liblinear", random_state=123).fit(X_train, y_train)
y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)


from sklearn.metrics import accuracy_score
print('Accuracy: ', accuracy_score(y_test, y_pred))


from sklearn. metrics import classification_report
print(classification_report(y_test, y_pred))


from sklearn.metrics import precision_score, recall_score
print("Overall Precision:",precision_score(y_test, y_pred))
print("Overall Recall:",recall_score(y_test, y_pred))


from sklearn.metrics import confusion_matrix
# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
#AUC = 0.82


from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#%%         Random Forest
#---------------------------------------------------------------
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=123).fit(X_train, y_train)

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)
cm = confusion_matrix(y_test, y_pred)
print ('Confusion Matrix:\n', cm, '\n')
print('Accuracy:', accuracy_score(y_test, y_pred))
print("Overall Precision:", precision_score(y_test, y_pred))
print("Overall Recall:", recall_score(y_test, y_pred))
auc = roc_auc_score(y_test, y_scores[:, 1])
print('\nAUC: ' + str(auc))

# calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1])

# plot ROC curve
fig = plt.figure(figsize=(6, 6))
# Plot the diagonal 50% line
plt.plot([0, 1], [0, 1], 'k--')
# Plot the FPR and TPR achieved by our model
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#%%         Final model
#---------------------------------------------------------------

X_train_final = df[df[TrainTest] == 'Train'][features].values #.values converts into n-dimensional numpy array
X_test_final = df[df[TrainTest] == 'Test'][features].values
y_train_final = df[df[TrainTest] == 'Train'][label].values
y_test_final = df[df[TrainTest] == 'Test'][label].values
#To use sklearn, these needs to be np arrays instead of dataframes

model_final = LogisticRegression(C=1/reg, solver="liblinear", random_state=123).fit(X_train_final, y_train_final)
#model_final = RandomForestClassifier(n_estimators=100, random_state=123).fit(X_train_final, y_train_final)
y_pred_final = model_final.predict(X_test_final).astype(int)
#Make sure the prediction is in datatype integer, otherwise Kaggle will not recognize it

passengerId = pd.Series(df[df[TrainTest] == 'Test']['PassengerId'].values, name='PassengerId')
prediction = pd.Series(y_pred_final, name='Survived')

df_final = pd.concat([passengerId, prediction], axis=1)
export_file = pd.DataFrame(data=df_final)
export_file.to_csv('titanic_predictions.csv', index=False)
