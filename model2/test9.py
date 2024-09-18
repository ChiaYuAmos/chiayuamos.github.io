import pandas as pd
import numpy as np

# 讀取原始特徵數據集
features_df = pd.read_csv('Phishing_Dataset8_features.csv')

# 讀取含有真偽標籤的數據集
data = pd.read_csv('Phishing_Dataset.csv', encoding='utf-8')

# 檢查兩個數據集中是否有對應的 url
assert len(features_df) == len(data), "數據集長度不一致"

# 將 url 列加回到 features_df 中
features_df['url'] = data['url']

# 根據真偽標籤給予 feedback 值，真的給 10.0，假的給 0.0
features_df['feedback'] = data['label'].apply(lambda x: 10.0 if x == 'benign' else 0.0)

# 調整欄位順序，把 url 移到 url_length 前面
columns = ['url'] + [col for col in features_df.columns if col != 'url']
features_df = features_df[columns]

# 儲存更新後的特徵數據集
features_df.to_csv('Phishing_Dataset8_features_updated.csv', index=False)

print("已將 url 移到 url_length 前面，並保存為 Phishing_Dataset8_features_updated.csv")