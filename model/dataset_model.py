import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Import Dataset
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split Dataset into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [100, 200, 300, 400, 500],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=0)
random_search_rf = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rf, n_iter=100, cv=5, n_jobs=-1, verbose=2, random_state=0)
random_search_rf.fit(X_train, y_train)
best_params_rf = random_search_rf.best_params_
print("Best parameters found for Random Forest: ", best_params_rf)
best_rf_classifier = random_search_rf.best_estimator_
best_rf_classifier.fit(X_train, y_train)

# Hyperparameter Tuning for Gradient Boosting
param_grid_gb = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb = GradientBoostingClassifier(random_state=0)
grid_search_gb = GridSearchCV(estimator=gb, param_grid=param_grid_gb, cv=5, n_jobs=-1, verbose=2)
grid_search_gb.fit(X_train, y_train)
best_params_gb = grid_search_gb.best_params_
print("Best parameters found for Gradient Boosting: ", best_params_gb)
best_gb_classifier = grid_search_gb.best_estimator_
best_gb_classifier.fit(X_train, y_train)

# Feature Importance from Random Forest
importances_rf = best_rf_classifier.feature_importances_
indices_rf = np.argsort(importances_rf)[-10:]  # Select top 10 features

# Transform the dataset to keep only the top 10 features
X_selected_rf = X[:, indices_rf]

# Split the dataset again with selected features
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_selected_rf, y, test_size=0.5, random_state=0)
X_train_rf = sc.fit_transform(X_train_rf)
X_test_rf = sc.transform(X_test_rf)

# Train the model with selected features
best_rf_classifier.fit(X_train_rf, y_train_rf)

# Feature Importance from Gradient Boosting
importances_gb = best_gb_classifier.feature_importances_
indices_gb = np.argsort(importances_gb)[-10:]  # Select top 10 features

# Transform the dataset to keep only the top 10 features
X_selected_gb = X[:, indices_gb]

# Split the dataset again with selected features
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(X_selected_gb, y, test_size=0.25, random_state=0)
X_train_gb = sc.fit_transform(X_train_gb)
X_test_gb = sc.transform(X_test_gb)

# Train the model with selected features
best_gb_classifier.fit(X_train_gb, y_train_gb)

# Model Evaluation for Random Forest
y_pred_rf = best_rf_classifier.predict(X_test_rf)
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
precision_rf = precision_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)

print("Random Forest Metrics:")
print("Confusion Matrix:\n", cm_rf)
print("Accuracy: {:.2f}%".format(accuracy_rf * 100))
print("Precision: {:.2f}%".format(precision_rf * 100))
print("Recall: {:.2f}%".format(recall_rf * 100))
print("F1 Score: {:.2f}%".format(f1_rf * 100))

# Model Evaluation for Gradient Boosting
y_pred_gb = best_gb_classifier.predict(X_test_gb)
cm_gb = confusion_matrix(y_test_gb, y_pred_gb)
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)
precision_gb = precision_score(y_test_gb, y_pred_gb)
recall_gb = recall_score(y_test_gb, y_pred_gb)
f1_gb = f1_score(y_test_gb, y_pred_gb)

print("Gradient Boosting Metrics:")
print("Confusion Matrix:\n", cm_gb)
print("Accuracy: {:.2f}%".format(accuracy_gb * 100))
print("Precision: {:.2f}%".format(precision_gb * 100))
print("Recall: {:.2f}%".format(recall_gb * 100))
print("F1 Score: {:.2f}%".format(f1_gb * 100))

# Normalized Confusion Matrix for Random Forest
data_rf = confusion_matrix(y_test_rf, y_pred_rf, normalize='all')
df_cm_rf = pd.DataFrame(data_rf, columns=np.unique(y_test_rf), index=np.unique(y_test_rf))
df_cm_rf.index.name = 'Actual'
df_cm_rf.columns.name = 'Predicted'
plt.figure(figsize=(6,3))
sns.set(font_scale=1.4)
sns.heatmap(df_cm_rf, cmap="Blues", annot=True, annot_kws={"size": 10})
plt.title("Random Forest Confusion Matrix Heat Map\n")
plt.savefig('ConfusionMatrix_RF', dpi=300, bbox_inches='tight')

# Normalized Confusion Matrix for Gradient Boosting
data_gb = confusion_matrix(y_test_gb, y_pred_gb, normalize='all')
df_cm_gb = pd.DataFrame(data_gb, columns=np.unique(y_test_gb), index=np.unique(y_test_gb))
df_cm_gb.index.name = 'Actual'
df_cm_gb.columns.name = 'Predicted'
plt.figure(figsize=(6,3))
sns.set(font_scale=1.4)
sns.heatmap(df_cm_gb, cmap="Blues", annot=True, annot_kws={"size": 10})
plt.title("Gradient Boosting Confusion Matrix Heat Map\n")
plt.savefig('ConfusionMatrix_GB', dpi=300, bbox_inches='tight')


