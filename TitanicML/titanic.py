import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv(r'/Users/mohammedrumaan/Desktop/ML Assignment/titanic/train.csv')

dataset['Age'] = dataset['Age'].fillna(dataset['Age'].mean())

dataset['Name'] = dataset['Name'].apply(lambda x: len(str(x).split()))

columns = [ 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = dataset[columns]
y = dataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=251, random_state=42)

X= pd.get_dummies(X_train[columns])
x_test = pd.get_dummies(X_test[columns])

X.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)
print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

rf_clf = RandomForestClassifier(n_estimators=10,random_state=42)

rf_clf.fit(X,y_train)

y_pred=rf_clf.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Calculate recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Calculate F1-score
f1 = f1_score(y_test, y_pred)
print("F1-score:", f1)

# Calculate AUC-ROC
auc_roc = roc_auc_score(y_test, y_pred)
print("AUC-ROC:", auc_roc)
