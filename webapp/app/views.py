from app import app
from flask import render_template, request, flash
from flask import Flask, jsonify, make_response, redirect, url_for
from StravaEffort import StravaActivity
from StravaUser import StravaUser
from StravaModel import StravaModel
from StravaAPI import StravaAPI
from StravaDB import StravaDB
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
import requests
import pymongo
import json
from datetime import datetime
from bson.binary import Binary
import pickle


# CLIENT = pymongo.MongoClient()
# CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
DB = StravaDB()
# athlete = DB.athletes.find_one({'id': 4478600},{'model': 1})
# MODEL = pickle.loads(athlete['model'])

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    # get all users
    q = """ SELECT
                id, firstname, lastname, city, state
            FROM athletes;
        """
    athletes = DB.execute(q)
    # athletes = DB.athletes.find()[:]
    return render_template('home.html', athletes=athletes)

@app.route('/token_exchange', methods=['GET', 'POST'])
def token_exchange():
    code = request.args.get('code', None)
    api = StravaAPI()
    data = api.exchange_token(code)
    athlete_dict = data['athlete']
    athlete_dict['token'] = {'access_token': data['access_token'],
                             'token_type': data['token_type']}
    if DB.athletes.find_one({'id': athlete_dict['id']}) is None:
        DB.athletes.insert(athlete_dict)

    return redirect(url_for('index'))


@app.route('/fit', methods=['POST'])
def fit():
    # get the strava data if not already there

    # train a model on all of the data
    uid = int(request.form.get('userid', None))

    u = StravaUser(uid, get_streams=True)
    all_rides_df = u.make_df((0, 30))
    y = all_rides_df.pop('time_int')
    X = all_rides_df.values
    model = RandomForestRegressor(max_depth=8)
    # model = GradientBoostingRegressor()

    print 'Fitting model.......'
    model.fit(X, y)
    print 'Model fit!'
    binary_model = Binary(pickle.dumps(model, protocol=2))
    DB.athletes.update({'id': uid}, {'$set': {'model': binary_model}})


    # m = StravaModel(model)
    # for a in u.activities:
    #     # forecast, true, pred_time = m.predict_activity(a)

    #     # put the predicted data in the db and leave a timestamp
    #     DB.activities.update(
    #         {'id': a.id},
    #         {'$set': {'streams.predicted_time.data': np.cumsum(forecast).tolist(),
    #                   'predicted_moving_time': pred_time,
    #                   'date_predicted': datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%SZ')
    #                   }}
    #         )
    #     print 'stored', a.id
    return str(len(u.activities))

@app.route('/get-data', methods=['POST'])
def get_data():
    uid = request.form.get('userid', None)
    print uid
    u = StravaUser(int(uid))
    u.get_activities()

    return str(len(u.activities))

@app.route('/check', methods=['POST'])
def check():
    uid = request.form.get('userid', None)
    print 'checkid', uid

    # if the user has no activities, get them from Strava
    num_activities = DB.activities.find({'athlete.id': int(uid)}).count()
    print 'Number of activities: ', num_activities
    if num_activities == 0:
        return 'new'

    # query = {'athlete.id': int(uid),
    #          'predicted_moving_time': {'$exists': True}}
    # num_predictions = DB.activities.find(query).count()
    has_model = DB.athletes.find({'model': {'$exists': True}}).count()

    if has_model == 0:
        return 'predict'

    return 'good'

@app.route('/rides/<userid>', methods=['GET', 'POST'])
def rides(userid):

    print 'creating user'
    u = StravaUser(int(userid))

    return render_template(
        'rides.html',
        activities = u.activities,
        athlete = u.name)

@app.route('/change', methods=['POST'])
def change():

    aid = int(request.form.get('id', 0))
    print 'Getting activity'
    a = DB.activities.find_one({'id': aid})
    print 'Initializing activity'
    a = StravaActivity(a, get_streams=True)
    print 'Loading model'

    a.time.raw_data -= a.time.raw_data[0]
    a.distance.raw_data -= a.distance.raw_data[0]
    print 'Predicting'
    a.predict(MODEL)
    print 'Predicted'
    return jsonify(a.to_dict())

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route("/train/<userid>")
def train(userid):
    return render_template('train.html', userid=userid)

@app.route('/sleep', methods=['POST'])
def sleep():
    import time
    time.sleep(5)

    return 'Done!'

if __name__ == '__main__':
    pass