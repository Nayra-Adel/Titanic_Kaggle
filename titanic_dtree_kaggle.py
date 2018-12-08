import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import tree

# get titanic training & test file
titanic_train = pd.read_csv("input/train.csv")
titanic_test = pd.read_csv("input/test.csv")

# keep the PassengerId just to use it in submission :)
PassengerId = titanic_test.PassengerId.copy()

# Survived Feature
Y = titanic_train.Survived.copy()

# check missing values in train data set
print(titanic_train.isnull().sum())

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

# map the Embarked
train_data['Embarked'] = train_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
titanic_test['Embarked'] = titanic_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# map the sex
train_data.Sex = train_data.Sex.map({'male': 0, 'female': 1})
titanic_test.Sex = titanic_test.Sex.map({'male': 0, 'female': 1})

# map => child: 0, young: 1, adult: 2, middle age: 3, old age: 4
# Mapping Age in training data
train_data.loc[train_data['Age'] <= 12, 'Age'] = 0
train_data.loc[(train_data['Age'] > 12) & (train_data['Age'] <= 20), 'Age'] = 1
train_data.loc[(train_data['Age'] > 20) & (train_data['Age'] <= 40), 'Age'] = 2
train_data.loc[(train_data['Age'] > 40) & (train_data['Age'] <= 65), 'Age'] = 3
train_data.loc[train_data['Age'] > 65, 'Age'] = 4

# Mapping Age in test data
titanic_test.loc[titanic_test['Age'] <= 12, 'Age'] = 0
titanic_test.loc[(titanic_test['Age'] > 12) & (titanic_test['Age'] <= 20), 'Age'] = 1
titanic_test.loc[(titanic_test['Age'] > 20) & (titanic_test['Age'] <= 40), 'Age'] = 2
titanic_test.loc[(titanic_test['Age'] > 40) & (titanic_test['Age'] <= 65), 'Age'] = 3
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

# split the data to train & validate
X_train, X_valid, y_train, y_valid = train_test_split(train_data, Y, test_size=0.2, random_state=7)

tree1 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=3, min_samples_leaf=10)

tree1.fit(X_train, y_train)

print(round(tree1.score(X_train, y_train) * 100, 2))
print(round(tree1.score(X_valid, y_valid) * 100, 2))

# 79.425
predict = tree1.predict(titanic_test)
submission = pd.DataFrame({
    "PassengerId": PassengerId,
    "Survived": predict
})
submission.to_csv('AllFeatures_Except_Cabin_PassengerId_Name.csv', index=False)

'''
when i drop 'Fare' & 'Parch' still the same result
Training : 83.71
Validate : 81.56
Kaggle   : 79.425
'''