# use gradient boosting to predict the target variable

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# read in the data
df = pd.read_csv('diabetes.csv')

# split the data into training and test sets
X = df.drop('Outcome', axis=1)

y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# instantiate the model
gb = GradientBoostingClassifier()

# fit the model
gb.fit(X_train, y_train)

# make predictions
y_pred = gb.predict(X_test)

# evaluate with 2 decimal places and as a percentage
print('Accuracy: {:.2%}'.format(accuracy_score(y_test, y_pred)))


