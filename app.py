
from flask import Flask, render_template, request
from predict import main_fun
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    return str(main_fun(userText))


if __name__ == "__main__":
    app.run()
