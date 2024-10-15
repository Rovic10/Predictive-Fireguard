from flask import Flask

app = Flask(__name__)

@app.before_first_request
def before_first_request():
    print("This will run before the first request.")

@app.route('/')
def hello():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
