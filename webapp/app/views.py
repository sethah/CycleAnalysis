from app import app
from flask import render_template, request, flash
from flask import Flask, jsonify, make_response
from StravaEffort import StravaActivity
from StravaUser import StravaUser
from StravaModel import StravaModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import requests
import pymongo
import json


# CLIENT = pymongo.MongoClient()
CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
DB = CLIENT.strava

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return "Hello seth"

@app.route('/home')
def home():
    # get the strava data if not already there

    # train a model on all of the data
    name = 'Seth'
    u = StravaUser(name, get_streams=True)
    all_rides_df = u.make_df()
    y = all_rides_df.pop('time_int')
    X = all_rides_df.values
    # model = RandomForestRegressor(max_depth=8)
    model = GradientBoostingRegressor()
    model.fit(X, y)
    m = StravaModel(model)
    for a in u.activities:
        forecast, true, pred_time = m.predict_activity(a)
        DB.activities.update(
            {'id': a.id},
            {'$set': {'streams.predicted_time.data': np.cumsum(forecast).tolist(),
                    'predicted_moving_time': pred_time}}
            )
    return render_template('main.html')

@app.route('/rides', methods=['GET', 'POST'])
def rides():
    u = StravaUser('Seth')
    aid = 134934515
    # if 'id' in request.form:
    a = DB.activities.find({'id': request.form.get('id', aid)})[0]
    a = StravaActivity(a, get_streams=True)
    a.time.raw_data -= a.time.raw_data[0]
    a.distance.raw_data -= a.distance.raw_data[0]


    return render_template(
        'poly.html',
        activity = a,
        activities = u.activities)

@app.route('/change', methods=['POST'])
def change():
    aid = int(request.form.get('id', 0))
    a = DB.activities.find_one({'id': aid})
    a = StravaActivity(a, get_streams=True)
    a.time.raw_data -= a.time.raw_data[0]
    a.distance.raw_data -= a.distance.raw_data[0]
    print a.name, a.time.raw_data[-1]
    return jsonify(a.to_dict())

@app.route("/chart")
def simple():
    return render_template('chart.html')

if __name__ == '__main__':
    pass