import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. 加載數據
dataframe = pd.read_csv('phishing_url_website.csv')
# 2. 檢查數據
print(dataframe.head(10))
print(dataframe.info())
# 3. 準備數據
# 刪除不需要的列，假設 'URL', 'Domain', 'Title' 對模型無幫助
dataframe = dataframe.drop(columns=['URL', 'Domain', 'Title'])
# 確保數據中沒有空值
dataframe = dataframe.dropna()
# 將非數值特徵轉換為數值特徵
le = LabelEncoder()
dataframe['TLD'] = le.fit_transform(dataframe['TLD'])
# 獲取特徵和標籤
X = dataframe.drop(columns=['label'])
y = dataframe['label']
# 4. 分割訓練和測試數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 5. 選擇和訓練模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
# 6. 評估模型性能
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
