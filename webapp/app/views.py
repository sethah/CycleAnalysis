from app import app
from flask import render_template, request, flash
from flask import Flask, jsonify, make_response
from StravaEffort import StravaActivity
from StravaUser import StravaUser
from StravaModel import StravaModel
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import requests
import pymongo
import gpolyencode
import json

CLIENT = pymongo.MongoClient()
DB = CLIENT.mydb

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    # get the strava data if not already there

    # train a model on all of the data
    name = 'Seth'
    u = StravaUser(name)
    all_rides_df = u.make_df()
    y = all_rides_df.pop('time_int')
    X = all_rides_df.values
    model = RandomForestRegressor(max_depth=8)
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

@app.route("/simple.png")
def simple():
    import datetime
    import StringIO
    import random
 
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter
 
    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    png_output = StringIO.StringIO()
    canvas.print_png(png_output)
    response=make_response(png_output.getvalue())
    response.headers['Content-Type'] = 'image/png'
    return response

if __name__ == '__main__':
    pass