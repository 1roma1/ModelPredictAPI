from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

credit_scoring_model = joblib.load("models/credit_scoring_model.pk")
minsk_apartment_model = joblib.load("models/minsk_apartment_model.pk")


@app.route("/credit_scoring/predict", methods=["POST"])
def credit_scoring_predict():
    features = request.json
    df = pd.DataFrame.from_dict(dict(features), orient='index').transpose()      
    default = credit_scoring_model.predict(df)
    return jsonify({"default": int(default[0])})


@app.route("/minsk_apartment/predict", methods=["POST"])
def minsk_apartment_price_predict():
    features = request.json
    df = pd.DataFrame.from_dict(dict(features), orient='index').transpose()      
    price = minsk_apartment_model.predict(df)
    return jsonify({"price": int(price[0])})
