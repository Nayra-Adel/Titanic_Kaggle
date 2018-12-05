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

# keep the PassengerId just to use it in submission :)
PassengerId = titanic_test.PassengerId.copy()

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
    3- Drop "Cabin" (have many missing values).
'''

train_data = titanic_train

# Extract the number from Ticket feature
train_data['Ticket'] = train_data['Ticket'].str.extract('(\d+)', expand=False)
titanic_test['Ticket'] = titanic_test['Ticket'].str.extract('(\d+)', expand=False)

# fill the missing in the Age & Embarked & Ticket in training data
train_data.Age.fillna(titanic_train.Age.median(), inplace=True)
train_data.Embarked.fillna("S", inplace=True)
train_data.Ticket.fillna(233866, inplace=True)

# fill the missing in the Age & Fare with the median in test data
titanic_test.Age.fillna(titanic_test.Age.median(), inplace=True)
titanic_test.Fare.fillna(titanic_test.Fare.median(), inplace=True)

# create categorical variable
train_data = pd.get_dummies(train_data, columns=["Pclass"])
train_data = pd.get_dummies(train_data, columns=["Embarked"])
titanic_test = pd.get_dummies(titanic_test, columns=["Pclass"])
titanic_test = pd.get_dummies(titanic_test, columns=["Embarked"])

# map the sex
train_data.Sex = train_data.Sex.map({'male': 0, 'female': 1})
titanic_test.Sex = titanic_test.Sex.map({'male': 0, 'female': 1})

# Mapping Age in training data
'''
Converting Numerical Age to Categorical Variable
    map:
        child: 0
        young: 1
        adult: 2 
        middle age: 3 
        old age: 4
'''
train_data.loc[train_data['Age'] <= 16, 'Age'] = 0
train_data.loc[(train_data['Age'] > 16) & (train_data['Age'] <= 35), 'Age'] = 1
train_data.loc[(train_data['Age'] > 35) & (train_data['Age'] <= 45), 'Age'] = 2
train_data.loc[(train_data['Age'] > 45) & (train_data['Age'] <= 65), 'Age'] = 3
train_data.loc[train_data['Age'] > 65, 'Age'] = 4

# Mapping Age in test data
titanic_test.loc[titanic_test['Age'] <= 16, 'Age'] = 0
titanic_test.loc[(titanic_test['Age'] > 16) & (titanic_test['Age'] <= 35), 'Age'] = 1
titanic_test.loc[(titanic_test['Age'] > 35) & (titanic_test['Age'] <= 45), 'Age'] = 2
titanic_test.loc[(titanic_test['Age'] > 45) & (titanic_test['Age'] <= 65), 'Age'] = 3
titanic_test.loc[titanic_test['Age'] > 65, 'Age'] = 4

# Mapping Fare in training data
fare_mean = train_data.Fare.mean()
fare_median = train_data.Fare.median()

train_data.loc[train_data['Fare'] <= (fare_mean/2), 'Fare'] = 0
train_data.loc[(train_data['Fare'] > (fare_mean/2)) & (train_data['Fare'] <= fare_mean), 'Fare'] = 1
train_data.loc[(train_data['Fare'] > fare_mean) & (train_data['Fare'] <= fare_median), 'Fare'] = 2
train_data.loc[train_data['Fare'] > fare_median, 'Fare'] = 3

# Mapping Fare in test data
titanic_test.loc[titanic_test['Fare'] <= (fare_mean/2), 'Fare'] = 0
titanic_test.loc[(titanic_test['Fare'] > (fare_mean/2)) & (titanic_test['Fare'] <= fare_mean), 'Fare'] = 1
titanic_test.loc[(titanic_test['Fare'] > fare_mean) & (titanic_test['Fare'] <= fare_median), 'Fare'] = 2
titanic_test.loc[titanic_test['Fare'] > fare_median, 'Fare'] = 3

# Drop unimportant features
train_data.drop(['Name', 'PassengerId', 'Survived', 'Cabin'], axis=1, inplace=True)
titanic_test.drop(['Name', 'PassengerId', 'Cabin'], axis=1, inplace=True)

Y = titanic_train.Survived.copy()

# split the data to train & validate
X_train, X_valid, y_train, y_valid = train_test_split(train_data, Y, test_size=0.2, random_state=7)

tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_leaf=20)

tree1.fit(X_train, y_train)

print(round(tree1.score(X_train, y_train) * 100, 2))
print(round(tree1.score(X_valid, y_valid) * 100, 2))

'''
when i drop Fare & Embarked_Q still the same result:

without Ticket Feature
max depth = 2
Training : 80.2
Validate : 72.63
Kaggle   : 76.555

max depth = 3 
Training : 81.6
Validate : 79.33
Kaggle   : 77.99

max depth = 4 
Training : 82.3
Validate : 76.54
Kaggle   : 77.99

with Ticket Feature:
max depth = 3
Training : 81.88
Validate : 79.89
Kaggle   : 78.947

Using Cabin feature is useless
'''

predict = tree1.predict(titanic_test)
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": predict
})
submission.to_csv('submission1.csv', index=False)