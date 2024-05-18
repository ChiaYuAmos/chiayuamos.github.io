import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 加載數據集
data = pd.read_csv('dataset_full.csv')

# 特徵和目標標籤
features = data.columns.difference(['phishing'])
X = data[features]
y = data['phishing']

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=200000)
# 訓練模型
model = RandomForestClassifier()
model.fit(X_train, y_train)
# 預測
y_pred = model.predict(X_test)
# 評估模型
print(classification_report(y_test, y_pred))
