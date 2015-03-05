from app import app
from flask import render_template, request, flash
from flask import Flask, jsonify, make_response, redirect, url_for
from StravaEffort import StravaActivity
from StravaUser import StravaUser
from StravaModel import StravaModel
from StravaAPI import StravaAPI
from StravaDB import StravaDB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_validation import cross_val_score, train_test_split
import numpy as np
import pandas as pd
import requests
import pymongo
import json
from datetime import datetime
from bson.binary import Binary
import pickle
import os.path
from werkzeug import secure_filename


# CLIENT = pymongo.MongoClient()
CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
MONGODB = CLIENT.strava
DB = StravaDB()

app.config['UPLOAD_FOLDER'] = 'app/uploads/'

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    DB = StravaDB()
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
    ath = data['athlete']
    ath['token'] = data['access_token']

    d = {'id': ath['id'],
         'firstname': ath['firstname'],
         'lastname': ath['lastname'],
         'sex': ath['sex'],
         'city': ath['city'],
         'state': ath['state'],
         'country': ath['country'],
         'access_token': ath['token']}
    DB.insert_values('athletes', d)
    DB.conn.commit()

    return redirect(url_for('index'))


@app.route('/fit', methods=['POST'])
def fit():
    # get the strava data if not already there

    # train a model on all of the data
    uid = int(request.form.get('userid', None))

    user = StravaUser(uid, get_streams=True, get_routes=False)

    indices = np.arange(len(user.activities))
    train_indices = np.random.choice(indices, size=int(len(user.activities)*0.75), replace=False)
    test_indices = np.setdiff1d(indices, train_indices)
    
    df = user.make_df(train_indices)
    y = df['velocity'].values
    cols = df.columns
    X = df[cols[np.where(cols!='velocity')]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    model = GradientBoostingRegressor(max_depth=3, min_samples_leaf=1000)
    print 'Fitting model.......'
    model.fit(X_train, y_train)
    print 'Model fit!'
    
    print 'Loading pickle'
    if os.path.isfile('model_%s.pkl' % user.userid):
        d = pickle.load(open('model_%s.pkl' % user.userid, 'rb'))
    else:
        d = {}

    print 'Dumping pickle'
    d[user.userid] = {'date': datetime.now(), 'model': model}
    pickle.dump(d, open('model_%s.pkl' % user.userid, 'wb'))
    print 'Pickle dumped'

    return str(len(user.activities))

@app.route('/upload', methods=['POST'])
def upload_gpx():
    uid = int(request.form.get('athlete_id', 0))
    ride_name = request.form.get('ride_title', 'New Route')
    if ride_name.strip() == '':
        ride_name = 'New Route'
    f = request.files['file']

    if f:
         # Make the filename safe, remove unsupported chars
        filename = secure_filename(f.filename)
        fpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(os.path.abspath(fpath))
        DB = StravaDB()
        DB.create_route(fpath, uid, ride_name)
    return redirect(url_for('rides', userid=4478600))

@app.route('/delete/route', methods=['POST'])
def delete_route():
    route_id = int(request.form.get('route_id', 0))
    athlete_id = int(request.form.get('athlete_id', 0))

    DB = StravaDB()
    q = """ DELETE FROM streams
            WHERE activity_id = %s
            AND athlete_id = %s
        """ % (route_id, athlete_id)
    DB.cur.execute(q)

    q = """ DELETE FROM routes
            WHERE id = %s
            AND athlete_id = %s
        """ % (route_id, athlete_id)
    DB.cur.execute(q)

    DB.conn.commit()

    return ''

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
    DB = StravaDB()
    q = """SELECT COUNT(*) FROM activities WHERE athlete_id = %s""" % uid
    DB.cur.execute(q)
    num_activities = DB.cur.fetchone()[0]
    print 'Number of activities: ', num_activities
    if num_activities == 0:
        return 'new'

    # query = {'athlete.id': int(uid),
    #          'predicted_moving_time': {'$exists': True}}
    # num_predictions = DB.activities.find(query).count()
    has_model = os.path.isfile('model_%s.pkl' % uid)
    print 'Has model', has_model

    if has_model == 0:
        return 'predict'

    return 'good'

@app.route('/rides/<userid>', methods=['GET', 'POST'])
def rides(userid):

    print 'creating user'
    u = StravaUser(int(userid))
    activities = []
    routes = []
    for a in u.activities:
        if a.is_route:
            routes.append(a)
        else:
            activities.append(a)

    # pass a single activity with all the streams
    activity = u.activities[0]
    activity.init_streams()
    d = pickle.load(open('model_%s.pkl' % u.userid, 'rb'))
    activity.predict(d[u.userid]['model'])

    return render_template(
        'rides.html',
        athlete = u,
        activities=activities,
        routes=routes,
        activity = activity)

@app.route('/change', methods=['POST'])
def change():

    aid = int(request.form.get('activity_id', 0))
    uid = int(request.form.get('athlete_id', 0))
    print 'Initializing activity'
    # TODO: FIX THIS AWFUL HACKY SHIT
    if aid < 10000:
        a = StravaActivity(aid, uid, get_streams=True, is_route=True)
    else:
        a = StravaActivity(aid, uid, get_streams=True)
    print 'Loading model'
    d = pickle.load(open('model_%s.pkl' % uid, 'rb'))
    print d
    print 'Predicting'
    a.predict(d[uid]['model'])
    print 'Predicted'
    print 'max is ', a.df.altitude.max()
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