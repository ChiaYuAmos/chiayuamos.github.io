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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=200)

# 訓練模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 評估模型
print(classification_report(y_test, y_pred))

# 自己的網址特徵提取函數
def extract_features(url):
    features = {}
    features['qty_dot_url'] = url.count('.')
    features['qty_hyphen_url'] = url.count('-')
    features['qty_underline_url'] = url.count('_')
    features['qty_slash_url'] = url.count('/')
    features['qty_dot_domain'] = url.split('//')[-1].split('/')[0].count('.')
    features['qty_hyphen_domain'] = url.split('//')[-1].split('/')[0].count('-')
    features['qty_vowels_domain'] = sum(map(url.count, "aeiou"))
    features['domain_length'] = len(url.split('//')[-1].split('/')[0])
    # 確保特徵名稱匹配
    for feature in X.columns:
        if feature not in features:
            features[feature] = 0  # 設定缺失的特徵為0
    return pd.DataFrame([features])

# 測試自己的網址
while True:
    test_url = input("輸入想要測試網址（exit離開）： ")
    if test_url.lower() == "exit":
        break
    test_features = extract_features(test_url)
    print(test_features)
    # 確保特徵數據框的列順序與訓練數據一致
    test_features = test_features[X.columns]
    # 預測
    prediction = model.predict(test_features)
    print(f"The URL '{test_url}' is predicted as: {'Phishing' if prediction[0] == 1 else 'Not Phishing'}")
