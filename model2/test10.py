import pandas as pd
import numpy as np
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from bs4 import BeautifulSoup
import requests
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# 讀取數據
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')

# 檢查缺失值
print("數據集中的缺失值:")
print(data.isnull().sum())

# 特徵提取函數
def extract_features(url):
    parsed_url = urlparse(url)
    features = {
        'url_length': len(url),
        'domain_length': len(parsed_url.netloc),
        'tld': parsed_url.netloc.split('.')[-1] if '.' in parsed_url.netloc else '',
        'has_ip': int(parsed_url.netloc.replace('.', '').isdigit()),  # 判斷是否有 IP 地址
        'num_special_chars': sum(not c.isalnum() and c not in ['.', ':', '/'] for c in url),
        'has_https': int(parsed_url.scheme == 'https'),
        'has_www': int('www.' in parsed_url.netloc),
        'pct_ext_hyperlinks': 0.0,
        'pct_ext_resource_urls': 0.0,
        'ext_favicon': 0.0,
        'feedback': 5.0,
    }
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            features.update({
                'pct_ext_hyperlinks': len(soup.find_all('a', href=True)) / (len(soup.find_all('a')) + 1),
                'pct_ext_resource_urls': len(soup.find_all(src=True)) / (len(soup.find_all()) + 1),
                'ext_favicon': int(bool(soup.find('link', rel='icon', href=True))),
            })
    except requests.exceptions.RequestException:
        pass
    return features

# 讀取已提取的特徵數據框
features_df = pd.read_csv('Phishing_Dataset10_features.csv')

# 處理 TLD 特徵（使用 One-Hot Encoding）
features_df = pd.get_dummies(features_df, columns=['tld'])

# 準備數據
X = features_df
y = data['label'].apply(lambda x: 1 if x == 'malicious' else 0)  # 轉換標籤為二元格式

# 將數據分為數值型和分類型欄位
numeric_cols = X.select_dtypes(include=[np.number]).columns
categorical_cols = X.select_dtypes(exclude=[np.number]).columns

# 將布爾類型轉換為整數
X[categorical_cols] = X[categorical_cols].apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

# 定義填補器
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# 填補數值型數據
X_numeric = pd.DataFrame(numeric_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

# 填補分類型數據
X_categorical = pd.DataFrame(categorical_imputer.fit_transform(X[categorical_cols]), columns=categorical_cols)

# 重組數據
X = pd.concat([X_numeric, X_categorical], axis=1)

# 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 標準化數值型數據
scaler = StandardScaler()
X_train_numeric = pd.DataFrame(scaler.fit_transform(X_train[numeric_cols]), columns=numeric_cols)
X_test_numeric = pd.DataFrame(scaler.transform(X_test[numeric_cols]), columns=numeric_cols)

# 重組經標準化的數據與分類數據
X_train = pd.concat([X_train_numeric, X_train[categorical_cols].reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_numeric, X_test[categorical_cols].reset_index(drop=True)], axis=1)

# 應用 SMOTE 處理類別不平衡
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 建立 MLP 模型
model = Sequential()
model.add(Dense(256, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# 編譯模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# 訓練模型
history = model.fit(X_resampled, y_resampled, epochs=1, batch_size=32, validation_split=0.2, verbose=1)

# 預測
print(model.predict(X_test))
print(model.predict(X_train))
y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()


# 評估模型
print("混淆矩陣:")
print(confusion_matrix(y_test, y_pred))
print("\n分類報告:")
print(classification_report(y_test, y_pred))
quit()

def scale_prediction(prediction):
    return prediction * 10.0

# 預測新的網址
def predict_url(model, url, scaler):
    features = extract_features(url)
    features_df = pd.DataFrame([features])
    features_df = pd.get_dummies(features_df, columns=['tld'])
    features_df = features_df.reindex(columns=X.columns, fill_value=0)
    features_numeric = scaler.transform(features_df[numeric_cols])
    features_combined = pd.concat([pd.DataFrame(features_numeric, columns=numeric_cols), features_df[categorical_cols]], axis=1)
    raw_prediction = model.predict(features_combined)[0][0]  # 获取原始预测分数
    scaled_prediction = scale_prediction(raw_prediction)  # 缩放到0.0-10.0
    if np.isnan(scaled_prediction):
        scaled_prediction = 0.0  # 处理NaN情况
    return features, scaled_prediction

# 測試網址
with open("0_testwebsite", 'r', encoding='utf-8') as file:
    test_urls = file.read().split("\n")

# 過濾無效的URL
urls_to_test = [url for url in test_urls if url and url not in ['True Website', 'Phishing Website', '']]

# 打印預測結果
results = []
for url in urls_to_test:
    features, prediction = predict_url(model, url, scaler)
    result = {
        'url': url,
        'prediction': 'malicious' if prediction < 5.0 else 'benign',
        'score': prediction
    }
    result.update(features)
    results.append(result)
    print(f"URL: {url}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {features}\n")

# 將結果存儲到 CSV
results_df = pd.DataFrame(results)
results_df.to_csv('Phishing_Dataset7_result10.csv', index=False, encoding='utf-8')

print("True Website:\n", "\t\n".join([r['url'] for r in results if r['prediction'] == 'benign']))
print("\nPhishing Website:\n", "\t\n".join([r['url'] for r in results if r['prediction'] == 'malicious']))

def adjust_feedback(existing_feedback, adjustment):
    # 控制feedback的最大/最小值範圍，避免超出邊界
    max_feedback = 10.0
    min_feedback = 0.0
    
    # 逐步減少調整幅度，根據當前feedback的距離邊界的程度來調整
    if adjustment > 0:
        # 如果反饋是正向的，即用戶認為是安全網站
        new_feedback = min(existing_feedback + adjustment, max_feedback)
    else:
        # 如果反饋是負向的，即用戶認為是釣魚網站
        new_feedback = max(existing_feedback + adjustment, min_feedback)

    return new_feedback

# 預測函數，檢查URL是否已在結果文件中存在
def predict_url_2(model, url, scaler):
    results_df = pd.read_csv('Phishing_Dataset7_result10.csv')
    if url in results_df['url'].values:
        print("URL已存在，根據現有預測結果。")
        existing_result = results_df.loc[results_df['url'] == url].iloc[0].to_dict()
        print(f"當前預測：{existing_result['prediction']}, 分數：{existing_result['score']}")
        
        # 使用者提供反饋
        input_feedback = input("這是釣魚網站嗎？(Yes = 1, No = 2, Nothing = 3): ")
        if input_feedback.isdigit():
            input_feedback = int(input_feedback)
            if input_feedback == 1:
                # 用戶反饋為釣魚網站，減少 feedback 權重
                new_feedback = adjust_feedback(existing_result['feedback'], -0.5)
                results_df.loc[results_df['url'] == url, 'feedback'] = new_feedback
            elif input_feedback == 2:
                # 用戶反饋為安全網站，增加 feedback 權重
                new_feedback = adjust_feedback(existing_result['feedback'], 0.5)
                results_df.loc[results_df['url'] == url, 'feedback'] = new_feedback

            # 使用更新的 feedback 欄位重新進行預測
            features = extract_features(url)
            features['feedback'] = new_feedback  # 更新的 feedback

            features_df = pd.DataFrame([features])
            features_df = pd.get_dummies(features_df, columns=['tld'])
            features_df = features_df.reindex(columns=X.columns, fill_value=0)

            # 標準化數據並進行預測
            features_numeric = scaler.transform(features_df[numeric_cols])
            features_combined = pd.concat([pd.DataFrame(features_numeric, columns=numeric_cols), features_df[categorical_cols]], axis=1)
            raw_prediction = model.predict(features_combined)[0][0]
            scaled_prediction = scale_prediction(raw_prediction)
            if np.isnan(scaled_prediction):
                scaled_prediction = 0.0

            # 更新新的預測結果
            results_df.loc[results_df['url'] == url, 'score'] = scaled_prediction
            results_df.loc[results_df['url'] == url, 'prediction'] = 'malicious' if scaled_prediction < 5.0 else 'benign'
            
            # 保存更新後的 CSV
            results_df.to_csv('Phishing_Dataset7_result10.csv', index=False, encoding='utf-8')

        # 返回最新的預測結果
        return results_df.loc[results_df['url'] == url].to_dict(orient='records')[0]
    
    else:
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        features_df = pd.get_dummies(features_df, columns=['tld'])
        features_df = features_df.reindex(columns=X.columns, fill_value=0)
        features_numeric = scaler.transform(features_df[numeric_cols])
        features_combined = pd.concat([pd.DataFrame(features_numeric, columns=numeric_cols), features_df[categorical_cols]], axis=1)
        raw_prediction = model.predict(features_combined)[0][0]
        scaled_prediction = scale_prediction(raw_prediction)
        if np.isnan(scaled_prediction):
            scaled_prediction = 0.0

        result = {
            'url': url,
            'prediction': 'malicious' if scaled_prediction < 5.0 else 'benign',
            'score': scaled_prediction
        }
        result.update(features)

        # Append the new result to results_df
        results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)
        results_df.to_csv('Phishing_Dataset7_result10.csv', index=False, encoding='utf-8')
        return result

while(1):
    input_url = input("Input your url(input exit to quit): ")
    if input_url == "exit":
        break
    result = predict_url_2(model, input_url, scaler)  # 直接取得結果字典
    print(f"URL: {result['url']}\nPrediction: {result['prediction']}\nScore: {result['score']}\nFeatures: {result}\n")

