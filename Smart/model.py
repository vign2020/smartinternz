import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew


from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv('Milk Grading.csv')

df2 = df.copy()
df2['Grade'] = df2['Grade'].replace({0.5: 1, 1: 1})
df2['Grade'] = df2['Grade'].astype(int)

df3 = df2.copy()
df3.drop('Turbidity', axis=1, inplace=True)


y = df3['Grade']
X = df3.drop('Grade', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


dt_classifier = tree.DecisionTreeClassifier()

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
