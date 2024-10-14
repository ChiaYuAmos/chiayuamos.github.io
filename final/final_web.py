import os
import re
import time
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau

import http.server
import socketserver

# 載入模型
model = load_model("final_model.keras")
print("模型已成功載入")
# 載入標準化器
scaler = joblib.load("final_model.pkl")
X = joblib.load("final_X.pkl")
print("標準化器已成功載入")

def extract_features(url):
    try:
        print(f"Processing URL: {url}")  
        parsed_url = urlparse(url)
        hostname = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        tld = hostname.split('.')[-1] if '.' in hostname else ''
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
            'num_dots': hostname.count('.'),
            'subdomain_level': hostname.count('.') - 1,
            'path_level': path.count('/'),
            'num_dash': url.count('-'),
            'num_dash_in_hostname': hostname.count('-'),
            'at_symbol': int('@' in url),
            'tilde_symbol': int('~' in url),
            'num_underscore': url.count('_'),
            'num_percent': url.count('%'),
            'num_query_components': query.count('&') + 1 if query else 0,
            'num_ampersand': url.count('&'),
            'num_hash': url.count('#'),
            'num_numeric_chars': sum(c.isdigit() for c in url),
            'no_https': int(parsed_url.scheme != 'https'),
            'random_string': int(bool(re.search(r'\b[a-zA-Z]{10,}\b', hostname))),
            'ip_address': int(bool(re.match(r'\b\d{1,3}(\.\d{1,3}){3}\b', hostname))),
            'domain_in_subdomains': int(any(tld in hostname for tld in ['com', 'org', 'net', 'edu', 'gov'])),
            'domain_in_paths': int(any(tld in path for tld in ['com', 'org', 'net', 'edu', 'gov'])),
            'https_in_hostname': int('https' in hostname),
            'hostname_length': len(hostname),
            'path_length': len(path),
            'query_length': len(query),
            'double_slash_in_path': int('//' in path),
            'num_sensitive_words': sum(word in url for word in ['secure', 'account', 'update', 'login']),
            'embedded_brand_name': int(any(brand in hostname for brand in ['facebook', 'google', 'paypal'])),
            'insecure_forms': 0,
            'relative_form_action': 0,
            'ext_form_action': 0,
            'abnormal_form_action': 0,
            # 'feedback':5.0
        }

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                features.update({
                    'pct_ext_hyperlinks': len(soup.find_all('a', href=True)) / (len(soup.find_all('a')) + 1),
                    'pct_ext_resource_urls': len(soup.find_all(src=True)) / (len(soup.find_all()) + 1),
                    'ext_favicon': int(bool(soup.find('link', rel='icon', href=True))),
                    'insecure_forms': int(any(form['action'].startswith('http:') for form in soup.find_all('form', action=True))),
                    'relative_form_action': int(any(form['action'].startswith('/') for form in soup.find_all('form', action=True))),
                    'ext_form_action': int(any(not form['action'].startswith(url) for form in soup.find_all('form', action=True))),
                    'abnormal_form_action': int(any(form['action'] in ['#', 'about:blank', 'javascript:true'] for form in soup.find_all('form', action=True))),
                    # 更多 HTML 相關特徵可以在這裡添加
                })
        except requests.exceptions.RequestException:
            pass
        # print(f"Finished processing URL: {url}\n{features}") 
        return features
    
    except Exception as e:
        print(f"Error processing URL {url}: {e}")
        return {}
# 特徵提取並建立特徵數據框

def scale_prediction(prediction):
    return prediction * 10.0
def adjust_feedback(existing_feedback, adjustment):
    # 控制feedback的最大/最小值範圍，避免超出邊界
    max_feedback = 10.0
    min_feedback = 0.0
    
    # 根據當前feedback的距離邊界的程度進行調整
    if adjustment > 0:
        # 如果反饋是正向的，即用戶認為是安全網站
        new_feedback = min(existing_feedback + adjustment, max_feedback)
    else:
        # 如果反饋是負向的，即用戶認為是釣魚網站
        new_feedback = max(existing_feedback + adjustment, min_feedback)
    return new_feedback

def save_to_results_csv(url, label, result):
    file_path = 'Phishing_Dataset2.csv'
    new_id = 0
    try:
        # 檢查文件是否存在
        if os.path.exists(file_path):
            results_df = pd.read_csv(file_path)
        else:
            # 如果文件不存在，初始化 DataFrame 並添加第一行空白
            results_df = pd.DataFrame(columns=["id", "url", "label", "result"])
            empty_row = {"id": "", "url": "", "label": "", "result": ""}
            results_df = pd.concat([pd.DataFrame([empty_row]), results_df], ignore_index=True)
        
        # 檢查 URL 是否已存在，存在則更新
        if url in results_df['url'].values:
            results_df.loc[results_df['url'] == url, ['label', 'result']] = [label, result]
        else:
            # 確定新的 ID
            if not results_df.iloc[0:].empty:
                max_id = results_df.iloc[0:]['id'].dropna().astype(int).max()
                new_id = max_id + 1
            else:
                new_id = 0
            # 新增記錄
            new_row = pd.DataFrame(
                [{"id": new_id, "url": url, "label": label, "result": result}]
            )
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        
        # 儲存回文件，確保 ID 保持為整數
        results_df.to_csv(file_path, index=False, encoding='utf-8')
        
        # print(f"已成功寫入 URL: {url}")
        return new_id
    
    except Exception as e:
        print(f"保存到 CSV 時發生錯誤：{e}")

def predict_url_2(model, url, scaler, point):
    flag = False
    final_score = 0.0
    feedback = 0.0
    features = None  
    try:
        # 嘗試讀取現有的結果文件
        results_df = pd.read_csv('Phishing_final_webdat.csv')
        results_df2 = pd.read_csv('Phishing_Dataset2_features.csv')
        url_in_csv = url in results_df['url'].values
        
        if url_in_csv:
            # URL已存在，根據現有預測結果進行處理
            existing_result = results_df.loc[results_df['url'] == url].iloc[0].to_dict()
            
            # 使用者提供反饋
            input_feedback = point
            # print(type(input_feedback))
            if input_feedback == "1":
                # 用戶反饋為釣魚網站，減少 feedback 權重
                new_feedback = adjust_feedback(existing_result['feedback'], -0.5)
                # print(1)
            elif input_feedback == "2":
                # 用戶反饋為安全網站，增加 feedback 權重
                new_feedback = adjust_feedback(existing_result['feedback'], 0.5)
                # print(2)
            else:
                # 保持 feedback 不變
                new_feedback = existing_result['feedback']
                # print(3)
            
            # 更新 feedback 欄位
            results_df.loc[results_df['url'] == url, 'feedback'] = new_feedback

            # 提取特徵並進行重新預測
            features = extract_features(url)
            features_df = pd.DataFrame([features])
            features_df = pd.get_dummies(features_df, columns=['tld'])
            features_df = features_df.reindex(columns=X.columns, fill_value=0)  # 確保使用正確的特徵列
            features_scaled = scaler.transform(features_df)
            raw_prediction = model.predict(features_scaled)[0][0]
            scaled_prediction = scale_prediction(raw_prediction)
            final_score = scaled_prediction * 0.5 + new_feedback * 0.5

            if np.isnan(scaled_prediction):
                scaled_prediction = 0.0
            # 更新新的預測結果
            results_df.loc[results_df['url'] == url, 'score'] = final_score
            results_df.loc[results_df['url'] == url, 'prediction'] = 'malicious' if final_score < 5.0 else 'benign'
            result = results_df.loc[results_df['url'] == url].iloc[0].to_dict()

        else:
            # 如果 URL 不在 CSV 中，進行新預測
            features = extract_features(url)
            features_df = pd.DataFrame([features])
            features_df = pd.get_dummies(features_df, columns=['tld'])
            features_df = features_df.reindex(columns=X.columns, fill_value=0)
            features_scaled = scaler.transform(features_df)
            raw_prediction = model.predict(features_scaled)[0][0]
            scaled_prediction = scale_prediction(raw_prediction)

            if np.isnan(scaled_prediction):
                scaled_prediction = 0.0

            feedback = 5.0
            if ("http"or "https") not in url:
                scaled_prediction = 0.5*scaled_prediction
            final_score = scaled_prediction * 0.5 + feedback * 0.5
            result = {
                'url': url,
                'prediction': 'malicious' if final_score < 5.0 else 'benign',
                'score': final_score,
                'feedback': feedback
            }
            result.update(features)

            # 將新記錄加入到 DataFrame 中
            new_result_df = pd.DataFrame([result])
            results_df = pd.concat([results_df, new_result_df], ignore_index=True)
            new_result_df2 = pd.DataFrame([features])
            results_df2 = pd.concat([results_df2, new_result_df2], ignore_index=True)


    except FileNotFoundError:
        # 如果 CSV 文件不存在，創建新文件並寫入預測結果
        features = extract_features(url)
        features_df = pd.DataFrame([features])
        features_df = pd.get_dummies(features_df, columns=['tld'])
        features_df = features_df.reindex(columns=X.columns, fill_value=0)
        features_scaled = scaler.transform(features_df)
        raw_prediction = model.predict(features_scaled)[0][0]
        scaled_prediction = scale_prediction(raw_prediction)
        
        if np.isnan(scaled_prediction):
            scaled_prediction = 0.0
            
        feedback = 5.0
        final_score = scaled_prediction * 0.5 + feedback * 0.5
        result = {
            'url': url,
            'prediction': 'malicious' if final_score < 5.0 else 'benign',
            'score': final_score,
            'feedback': feedback
        }
        result.update(features)
        # 創建新的 DataFrame
        results_df = pd.DataFrame([result])
        results_df2 = pd.DataFrame([features])

    # 最後寫入或更新 CSV 文件
    results_df.to_csv('Phishing_final_webdat.csv', index=False, encoding='utf-8')
    results_df2.to_csv('Phishing_Dataset2_features.csv', index=False, encoding='utf-8')

    
    if final_score >= 5.0:
        flag = True
        new_id = save_to_results_csv(url, "benign", 0)
    else:
        new_id = save_to_results_csv(url, "malicious", 1)
    # print("\n\n",final_score,end="\n\n")
    if new_id%10==0:
            os.system("python3 final_model2.py")
            print("測試")
    return features, final_score, result, flag

PORT = 5050

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/index.css':
            # Serve the CSS file
            self.send_response(200)
            self.send_header("Content-type", "text/css")
            self.end_headers()
            with open("index.css", "rb") as css_file:
                self.wfile.write(css_file.read())
        
        elif self.path == "/tool.html":
            # Serve the tool.html page with a form
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            tool_page = """
            <!DOCTYPE html>
            <html lang="zh-Hant">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>網址分析工具</title>
                    <link rel="stylesheet" href="index.css">
                </head>
                <body>
                    <div class="container">
                        <h1>這是網址分析工具頁面</h1>
                        <div class="form-group">
                            <form method="POST" action="/tool.html">
                                請輸入網址：<input type="text" name="input_field" placeholder="在此輸入網址">
                                <button type="submit">送出</button><br>
                            </form>
                            <div class="description2" style="border: 1px solid #424242;">
                                使用此工具分析輸入的網址安全性<br>將顯示出有這個網址有多少的可信度<br>低於50%需要多加留意。
                            </div>
                        </div>
                    </div><br>
                    <a href="/">返回首頁</a>
                </body>
            </html>
            """
            self.wfile.write(tool_page.encode("utf-8"))
        
        elif self.path == "/report.html":
            # Serve the report.html page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            report_page = """
            <!DOCTYPE html>
            <html lang="zh-Hant">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>網址回饋工具</title>
                    <link rel="stylesheet" href="index.css">
                </head>
                <body>
                    <div class="container">
                        <h1>這是網址回報機制頁面</h1>
                        <div class="form-group">
                            <form method="POST" action="/report.html">
                                <label for="feedback_url">請輸入可疑網址：</label>
                                <input type="text" id="feedback_url" name="feedback_url" placeholder="在此輸入網址"><br><br>
                                
                                <label>請選擇此網址的狀態：</label><br>
                                <input type="radio" id="correct" name="url_status" value="correct">
                                <label for="correct">正確的網址</label><br>
                                
                                <input type="radio" id="suspicious" name="url_status" value="suspicious">
                                <label for="suspicious">可疑的網址</label><br><br>
                                
                                <button type="submit">送出</button>
                            </form>
                            <div class="description2" style="border: 1px solid #424242;">
                                回報可疑的網址以提升我們的資料庫安全性。
                            </div>
                        </div>
                    </div><br>
                    <a href="/">返回首頁</a>
                </body>
            </html>
            """
            self.wfile.write(report_page.encode("utf-8"))
        
        else:
            # Serve the main form page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html_content = """
            <!DOCTYPE html>
            <html lang="zh-Hant">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>簡單輸入表單</title>
                    <link rel="stylesheet" href="index.css">
                </head>
                <body>
                    <h1>惡意網址分析</h1>
                    <div class="container">
                        <div class="options">
                            <div class="option">
                                <a href="tool.html">
                                    <button class="option-button">網址分析工具</button>
                                </a>
                                <div class="description">使用此工具分析輸入的網址安全性。</div>
                            </div>
                            <div class="option">
                                <a href="report.html">
                                    <button class="option-button">網址回饋工具</button>
                                </a>
                                <div class="description">回報疑似惡意或釣魚網站。</div>
                            </div>
                        </div>
                    </div>
                </body>
            </html>
            """
            self.wfile.write(html_content.encode("utf-8"))

    def do_POST(self):
        # Handle form data submission
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        form_data = parse_qs(post_data)

        if self.path == "/tool.html":
            # Handle input from tool.html
            input_url = form_data.get('input_field', [''])[0]
            if input_url == "exit":
                quit()
            try:
                features, prediction, result, flag = predict_url_2(model, input_url, scaler, 3)
            except Exception as e:
                flag, prediction = "Error in prediction", str(e)
            # print(prediction,end="\n\n")

            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            response_content = f"""
            <!DOCTYPE html>
            <html lang="zh-Hant">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>結果</title>
                    <link rel="stylesheet" href="index.css">
                </head>
                <body>
                    <div class="container">
                        <h1>你輸入的網址是: {input_url}</h1>
                        <h1>這網址是有 {format(float(prediction)*10.0, ".0f") if isinstance(prediction, (int, float)) else prediction}%為真實網址</h1>
                        <a href="/" >返回</a>
                    </div>
                </body>
            </html>
            """
            self.wfile.write(response_content.encode("utf-8"))

        elif self.path == "/report.html":
            # Handle input from report.html
            feedback_url = form_data.get('feedback_url', [''])[0]
            url_status = form_data.get('url_status', [''])[0]  # 获取勾选的状态
            # Respond with the result page
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            # Define message based on url_status
            status_message = "正確的網址" if url_status == "correct" else "可疑的網址" if url_status == "suspicious" else "未知狀態"
            point =  "2" if url_status == "correct" else "1" if url_status == "suspicious" else "未知狀態"
            try:
                features, prediction, result, flag = predict_url_2(model, feedback_url, scaler, point)
                if "http" not in feedback_url:
                    prediction = 0.5*prediction
            except Exception as e:
                flag, prediction = "Error in prediction", str(e)
            response_content = f"""
            <!DOCTYPE html>
            <html lang="zh-Hant">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>回報確認</title>
                    <link rel="stylesheet" href="index.css">
                </head>
                <body>
                    <div class="container">
                        <h1>成功回報網址：{feedback_url}</h1>
                        <h1>這網址是有 {format(float(prediction)*10.0, ".0f") if isinstance(prediction, (int, float)) else prediction}%為真實網址</h1>
                        <p>您標記此網址為：{status_message}</p>
                        <p>感謝您協助我們改進安全性！</p>
                        <a href="/">返回</a>
                    </div>
                </body>
            </html>
            """
            self.wfile.write(response_content.encode("utf-8"))

# Start the server
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"伺服器正在執行於 http://localhost:{PORT}")
    httpd.serve_forever()