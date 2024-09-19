from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['input_field']
        return f'你輸入的是: {user_input}'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
