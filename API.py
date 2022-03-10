import joblib
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*" : {"origins": ":"}})

#Loading saved model
mod = joblib.load('lin_reg_2.pkl')

@app.route('/', methods=['GET'])
def home(): return '<h1>API is running</h1>'

@app.route('/predict', methods = ['GET'])
def predict():
    x = request.args['x']
    x = float(x)
    prediction = mod.predict([[x]])
    return {'prediction' : round(float(prediction),3)}

if __name__ == '__main__':
    app.run(port=8000, debug = True)
