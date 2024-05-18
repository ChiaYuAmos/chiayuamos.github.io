import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# import seaborn as sns 
import matplotlib.pyplot as plt 
data = pd.read_csv("phishing_url_website.csv")
# data.head()
# data.info()
object_cols = [i for i in data.columns.tolist() if data[i].dtype=="object"]
data.drop(object_cols, axis=1, inplace=True)
top_corr = data.corrwith(data["label"]).abs().sort_values(ascending=False)[:6].index.to_list()
# [data[i].value_counts() for i in data[top_corr]]
scaler = MinMaxScaler()
X = data.drop("label", axis=1)
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5, random_state=42)
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)
rfc_model = RandomForestClassifier()
rfc_model.fit(scaled_X_train, y_train)
rfc_preds = rfc_model.predict(scaled_X_test)
print(classification_report(y_test, rfc_preds))
