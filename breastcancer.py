import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#LOAD THE DATA
from google.colab import files
uploaded=files.upload()
df=pd.read_csv('data.csv')
df.head(10)
#COUNTING THE NUMBER OF ROWS AND COLUMNS IN THE DATA SET
df.shape
#COUNTING ALL THE MISSING VALUES
df.isna().sum()
df.dropna(axis=1)
df.shape
df['diagnosis'].value_counts()
sns.countplot(df['diagnosis'],label='count')
df.dtypes
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))
sns.pairplot(df.iloc[:,1:9])
df.iloc[:,1:12].corr()
plt.figsize=(10,10)
sns.heatmap(df.iloc[:,1:12].corr(),annot=True,fmt='.0%')
X=df.iloc[:,2:31].values
Y=df.iloc[:,1].values
type(X)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
def models(X_train,Y_train):
#USING LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state = 0)
log.fit(X_train, Y_train)
#USING KNEIGHBORSCLASSIFIER
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, Y_train)
#USING SVC LINEAR
from sklearn.svm import SVC
svc_lin = SVC(kernel = 'linear', random_state = 0)
svc_lin.fit(X_train, Y_train)
#USING SVC RBF
from sklearn.svm import SVC
svc_rbf = SVC(kernel = 'rbf', random_state = 0)
svc_rbf.fit(X_train, Y_train)
#USING GAUSSIANNB
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
gauss.fit(X_train, Y_train)
#USING DECISIONTREECLASSIFIER
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
tree.fit(X_train, Y_train)
#USING RANDOMFORESTCLASSIFIER METHOD OF ENSEMBLE CLASS
TO USE RANDOM FOREST CLASSIFICATION ALGORITHM
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest.fit(X_train, Y_train)
#PRINT MODEL ACCURACY ON THE TRAINING DATA. print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
print('[1]K Nearest Neighbor Training Accuracy:', knn.score(X_train, Y_train))
print('[2]Support Vector Machine (Linear Classifier) Training Accuracy:', svc_lin.score(X_train, Y_train))
print('[3]Support Vector Machine (RBF Classifier) Training Accuracy:', svc_rbf.score(X_train, Y_train))
print('[4]Gaussian Naive Bayes Training Accuracy:', gauss.score(X_train, Y_train))
print('[5]Decision Tree Classifier Training Accuracy:', tree.score(X_train,
Y_train))
print('[6]Random Forest Classifier Training Accuracy:', forest.score(X_train, Y_train))
return log, knn, svc_lin, svc_rbf, gauss, tree, forest
model = models(X_train,Y_train)
from sklearn.metrics import confusion_matrix
for i in range(len(model)):
cm = confusion_matrix(Y_test, model[i].predict(X_test))
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print(cm)
print('Model[{}] Testing Accuracy = "{}!"'.format(i, (TP + TN) / (TP + TN + FN
+ FP)))
print()
#WAYS TO GET THE CLASSIFICATION ACCURACY & OTHER METRICS
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
for i in range(len(model)):
print('Model ',i)
#Check precision, recall, f1-score
print( classification_report(Y_test, model[i].predict(X_test)) )
#Another way to get the models accuracy on the test data
print( accuracy_score(Y_test, model[i].predict(X_test)))
print()
#PRINT PREDICTION OF RANDOM FOREST CLASSIFIER MODEL
pred = model[6].predict(X_test)
print(pred)
print()
#PRINT THE ACTUAL VALUES
print(Y_test)
