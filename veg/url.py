import http.server
import socketserver
from urllib.parse import parse_qs
import final.share as share
PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # 回傳表單頁面
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
            <style>
                body {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                }
                .container {
                    text-align: center;
                }
                .form-group {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                h1 {
                    margin-right: 10px;
                }
                input {
                    padding: 5px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="form-group">
                    <h1>請輸入文字：</h1>
                    <form method="POST">
                        <input type="text" name="input_field" placeholder="在此輸入...">
                        <button type="submit">送出</button>
                    </form>
                </div>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html_content.encode("utf-8"))

    def do_POST(self):
        # 取得使用者提交的資料
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        form_data = parse_qs(post_data)

        # 獲取輸入的值
        user_input = form_data.get('input_field', [''])[0]
        share.shared_variable = user_input
        print(share.shared_variable)
        # 將使用者輸入的資料打印到終端機
        print(f"收到的輸入: {user_input}")

        # 回傳顯示使用者輸入的值
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        response = f"""
        <!DOCTYPE html>
        <html lang="zh-Hant">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>結果</title>
            <style>
                body {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    font-family: Arial, sans-serif;
                }}
                .container {{
                    text-align: center;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>你輸入的是: {user_input}</h1>
                <a href="/">返回</a>
            </div>
        </body>
        </html>
        """
        self.wfile.write(response.encode("utf-8"))

# 啟動伺服器
with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print(f"伺服器正在執行於 http://localhost:{PORT}")
    httpd.serve_forever()
