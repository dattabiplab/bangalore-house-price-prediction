
from flask import Flask, request, jsonify
from flask import render_template
import json
import pickle
import numpy as np
import os

app = Flask(__name__)

__locations = None
__data_columns = None
__model = None

# with open(os.path.join('artifacts', 'bangalore_home_prices_model.pickle'), 'rb') as f:
#     __model = pickle.load(f)

@app.route("/")
def html_Calling():
    return render_template('app.html')
@app.route('/api/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': get_location_names()
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response

@app.route('/api/predict_home_price',methods=['GET','POST'])
def predict_home_price():
    total_sqft = float(request.form['total_sqft'])
    location = request.form['location']
    bhk = int(request.form['bhk'])
    bath = int(request.form['bath'])
    response = jsonify({
        'estimated_price': get_estimated_price(location, total_sqft, bhk, bath)
    })
    response.headers.add('Access-Control-Allow-Origin','*')
    return response


def get_estimated_price(location, sqft, bhk, bath):
    global __data_columns
    global __model
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return round(__model.predict([x])[0], 2)


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations
    global __model

    try:
        with open("./artifacts/columns.json", "r") as f:
            __data_columns = json.load(f)['data_columns']
            __locations = __data_columns[3:]
        # print(f"Data columns: {__data_columns}")

        if __model is None:
            with open("./artifacts/bangalore_home_prices_model.pickle", 'rb') as f:
                __model = pickle.load(f)
        print("loading saved artifacts...done")
    except Exception as e:
        print(f"Error loading artifacts: {e}")



def get_location_names():
    global __locations
    return __locations


def get_data_columns():
    global __data_columns
    return __data_columns


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    load_saved_artifacts()
    app.run()
