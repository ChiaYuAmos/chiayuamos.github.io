import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 假設這個函數已經實現
def extract_features_from_url(url):
    # 實現特徵提取邏輯
    # 返回一個與數據集特徵相同的特徵向量
    return np.random.rand(X.shape[1])  # 示例返回隨機特徵

# 導入數據集
dataset = pd.read_csv('dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# 切分數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# 特徵縮放
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 隨機森林超參數調整
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
print("隨機森林最佳參數: ", best_params_rf)
best_rf_classifier = random_search_rf.best_estimator_
best_rf_classifier.fit(X_train, y_train)

# 梯度提升超參數調整
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
print("梯度提升最佳參數: ", best_params_gb)
best_gb_classifier = grid_search_gb.best_estimator_
best_gb_classifier.fit(X_train, y_train)

# 從隨機森林獲取特徵重要性
importances_rf = best_rf_classifier.feature_importances_
indices_rf = np.argsort(importances_rf)[-10:]  # 選擇前10個重要特徵

# 轉換數據集以僅保留前10個特徵
X_selected_rf = X[:, indices_rf]

# 重新切分數據集
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_selected_rf, y, test_size=0.5, random_state=0)
X_train_rf = sc.fit_transform(X_train_rf)
X_test_rf = sc.transform(X_test_rf)

# 使用選擇的特徵訓練模型
best_rf_classifier.fit(X_train_rf, y_train_rf)

# 從梯度提升獲取特徵重要性
importances_gb = best_gb_classifier.feature_importances_
indices_gb = np.argsort(importances_gb)[-10:]  # 選擇前10個重要特徵

# 轉換數據集以僅保留前10個特徵
X_selected_gb = X[:, indices_gb]

# 重新切分數據集
X_train_gb, X_test_gb, y_train_gb, y_test_gb = train_test_split(X_selected_gb, y, test_size=0.25, random_state=0)
X_train_gb = sc.fit_transform(X_train_gb)
X_test_gb = sc.transform(X_test_gb)

# 使用選擇的特徵訓練模型
best_gb_classifier.fit(X_train_gb, y_train_gb)

# 隨機森林模型評估
y_pred_rf = best_rf_classifier.predict(X_test_rf)
cm_rf = confusion_matrix(y_test_rf, y_pred_rf)
accuracy_rf = accuracy_score(y_test_rf, y_pred_rf)
precision_rf = precision_score(y_test_rf, y_pred_rf)
recall_rf = recall_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)

print("隨機森林評估指標:")
print("混淆矩陣:\n", cm_rf)
print("準確率: {:.2f}%".format(accuracy_rf * 100))
print("精確率: {:.2f}%".format(precision_rf * 100))
print("召回率: {:.2f}%".format(recall_rf * 100))
print("F1分數: {:.2f}%".format(f1_rf * 100))

# 梯度提升模型評估
y_pred_gb = best_gb_classifier.predict(X_test_gb)
cm_gb = confusion_matrix(y_test_gb, y_pred_gb)
accuracy_gb = accuracy_score(y_test_gb, y_pred_gb)
precision_gb = precision_score(y_test_gb, y_pred_gb)
recall_gb = recall_score(y_test_gb, y_pred_gb)
f1_gb = f1_score(y_test_gb, y_pred_gb)

print("梯度提升評估指標:")
print("混淆矩陣:\n", cm_gb)
print("準確率: {:.2f}%".format(accuracy_gb * 100))
print("精確率: {:.2f}%".format(precision_gb * 100))
print("召回率: {:.2f}%".format(recall_gb * 100))
print("F1分數: {:.2f}%".format(f1_gb * 100))

# 設置支持中文顯示的字體
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑體
plt.rcParams['axes.unicode_minus'] = False

# 隨機森林歸一化混淆矩陣
data_rf = confusion_matrix(y_test_rf, y_pred_rf, normalize='all')
df_cm_rf = pd.DataFrame(data_rf, columns=np.unique(y_test_rf), index=np.unique(y_test_rf))
df_cm_rf.index.name = '實際值'
df_cm_rf.columns.name = '預測值'
plt.figure(figsize=(6, 3))
sns.set(font_scale=1.4)
sns.heatmap(df_cm_rf, cmap="Blues", annot=True, annot_kws={"size": 10})
plt.title("隨機森林混淆矩陣熱圖\n")
plt.savefig('ConfusionMatrix_RF', dpi=300, bbox_inches='tight')

# 梯度提升歸一化混淆矩陣
data_gb = confusion_matrix(y_test_gb, y_pred_gb, normalize='all')
df_cm_gb = pd.DataFrame(data_gb, columns=np.unique(y_test_gb), index=np.unique(y_test_gb))
df_cm_gb.index.name = '實際值'
df_cm_gb.columns.name = '預測值'
plt.figure(figsize=(6, 3))
sns.set(font_scale=1.4)
sns.heatmap(df_cm_gb, cmap="Blues", annot=True, annot_kws={"size": 10})
plt.title("梯度提升混淆矩陣熱圖\n")
plt.savefig('ConfusionMatrix_GB', dpi=300, bbox_inches='tight')

# 自訂網址特徵提取和預測循環
while True:
    # Prompt for URL input
    url = input("Enter a URL to predict (type 'exit' to stop): ")
    
    if url.lower() == 'exit':
        print("Exiting...")
        break
    # Extract features from the URL
    features_custom_url = extract_features_from_url(url)
    features_custom_url = np.array(features_custom_url).reshape(1, -1)
    # Use selected features from Random Forest
    features_custom_url_rf = features_custom_url[:, indices_rf] if features_custom_url.ndim > 1 else features_custom_url[indices_rf].reshape(1, -1)
    features_custom_url_rf = sc.transform(features_custom_url_rf)
    # Predict using Random Forest model
    prediction_rf = best_rf_classifier.predict(features_custom_url_rf)
    # Output prediction
    if prediction_rf == 1:
        print(f"The URL '{url}' is predicted as phishing.")
    else:
        print(f"The URL '{url}' is predicted as legitimate.")