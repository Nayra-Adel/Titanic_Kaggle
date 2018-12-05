import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree

import seaborn as sns
import matplotlib.pyplot as plt

plt.rc("font", size=14)

sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# get titanic training & test file
titanic_train = pd.read_csv("input/train.csv")
titanic_test = pd.read_csv("input/test.csv")

# check missing values in train data set
print(titanic_train.isnull().sum())

# what the 'Age' variable looks like.
ax = titanic_train["Age"].hist(bins=15, color='teal', alpha=0.8)
ax.set(xlabel='Age', ylabel='Count')
plt.show()

# what the 'Fare' variable looks like.
ax = titanic_train["Fare"].hist(bins=50, color='teal', alpha=0.8)
ax.set(xlabel='Fare', ylabel='Count')
plt.show()

# most passengers boarded in Southampton (S)
sns.countplot(x='Embarked', data=titanic_train, palette='Set1')
plt.show()

'''
Process the Data:
    1- If "Age" is missing , put 28 (median age).
    2- If "Embark" is missing , put "S" (most passengers boarded in S).
    3- Drop The features "Ticket" and "Cabin" (have many missing values).
'''

train_data = titanic_train

# fill the missing in the Age & Embarked
train_data["Age"].fillna(titanic_train.Age.median(), inplace=True)
train_data["Embarked"].fillna("S", inplace=True)

# create categorical variable
train_data = pd.get_dummies(train_data, columns=["Pclass"])
train_data = pd.get_dummies(train_data, columns=["Embarked"])

# Mapping Fare

fare_mean = train_data.Fare.mean()
fare_median = train_data.Fare.median()

train_data.loc[train_data['Fare'] <= (fare_mean/2), 'Fare'] = 0
train_data.loc[(train_data['Fare'] > (fare_mean/2)) & (train_data['Fare'] <= fare_mean), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > fare_mean) & (train_data['Fare'] <= fare_median), 'Fare'] = 2
train_data.loc[train_data['Fare'] > fare_median, 'Fare'] = 3

# Mapping Age
train_data.loc[train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 32), 'Age'] = 1
train_data.loc[(train_data['Age'] > 32) & (train_data['Age'] <= 48), 'Age'] = 2
train_data.loc[(train_data['Age'] > 48) & (train_data['Age'] <= 64), 'Age'] = 3
train_data.loc[train_data['Age'] > 64, 'Age'] = 4

# map the sex
train_data.Sex = train_data.Sex.map({'male': 0, 'female': 1})
titanic_test.Sex = titanic_test.Sex.map({'male': 0, 'female': 1})

train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(train_data['Title'].value_counts())
train_data.Title = train_data.Title.map(
                                    dict(Mr=0, Miss=1, Mrs=2, Master=3,
                                         Dr=3, Rev=3, Col=3, Major=3, Mlle=3,
                                         Countess=3, Ms=3, Lady=3, Jonkheer=3,
                                         Don=3, Mme=3, Capt=3, Sir=3)
                                    )

titanic_test['Title'] = titanic_test['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(titanic_test['Title'].value_counts())
titanic_test.Title = titanic_test.Title.map(
                                    dict(Mr=0, Miss=1, Mrs=2, Master=3,
                                         Rev=3, Col=3, Dr=3, Dona=3, Ms=3)
                                    )
# Drop unimportant features
train_data.drop(['Name', 'Cabin', 'PassengerId', 'Survived', 'Ticket'], axis=1, inplace=True)

# ----
# Test
# ----

# keep the PassengerId just to use it in submission :)
PassengerId = titanic_test.PassengerId.copy()

# fill the missing in the Age & Fare with the median
titanic_test.Age.fillna(titanic_test.Age.median(), inplace=True)
titanic_test.Fare.fillna(titanic_test.Fare.median(), inplace=True)

# create categorical variable
titanic_test = pd.get_dummies(titanic_test, columns=["Pclass"])
titanic_test = pd.get_dummies(titanic_test, columns=["Embarked"])

# Mapping Fare
titanic_test.loc[titanic_test['Fare'] <= (fare_mean/2), 'Fare'] = 0
titanic_test.loc[(titanic_test['Fare'] > (fare_mean/2)) & (titanic_test['Fare'] <= fare_mean), 'Fare'] = 1
titanic_test.loc[(titanic_test['Fare'] > fare_mean) & (titanic_test['Fare'] <= fare_median), 'Fare'] = 2
titanic_test.loc[titanic_test['Fare'] > fare_median, 'Fare'] = 3

# Mapping Age
titanic_test.loc[titanic_test['Age'] <= 16, 'Age'] = 0
titanic_test.loc[(titanic_test['Age'] > 16) & (titanic_test['Age'] <= 32), 'Age'] = 1
titanic_test.loc[(titanic_test['Age'] > 32) & (titanic_test['Age'] <= 48), 'Age'] = 2
titanic_test.loc[(titanic_test['Age'] > 48) & (titanic_test['Age'] <= 64), 'Age'] = 3
titanic_test.loc[titanic_test['Age'] > 64, 'Age'] = 4

# Drop unimportant features
titanic_test.drop(['Name', 'Cabin', 'PassengerId', 'Ticket'], axis=1, inplace=True)

Y = titanic_train.Survived.copy()

# split the data to train & validate
X_train, X_valid, y_train, y_valid = train_test_split(train_data, Y, test_size=0.2, random_state=7)

tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_leaf=20)

tree1.fit(X_train, y_train)

print(round(tree1.score(X_train, y_train) * 100, 2))
print(round(tree1.score(X_valid, y_valid) * 100, 2))

'''
train_data.drop(['Fare', 'Embarked_Q'], axis=1, inplace=True)
titanic_test.drop(['Fare', 'Embarked_Q'], axis=1, inplace=True)

when i drop Fare & Embarked_Q still the same result:

max depth = 3 
Training : 81.6
Validate : 79.33
Kaggle   : 77.99

max depth = 1 || 2
Training : 80.2
Validate : 72.63
Kaggle   : 76.555

max depth = 4 || 8
Training : 82.3
Validate : 76.54
Kaggle   : 77.99

Extract Title feature from Name:
max depth = 3
Training : 82.3
Validate : 79.89
Kaggle   : 76.555

'''
predict = tree1.predict(titanic_test)

submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": predict
})
submission.to_csv('submission1.csv', index=False)