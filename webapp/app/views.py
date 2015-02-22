from app import app
from flask import render_template, request, flash
from flask import Flask
from StravaEffort import StravaActivity
import requests
import pymongo
import gpolyencode
import json

CLIENT = pymongo.MongoClient()
DB = CLIENT.mydb

@app.route("/test")
def ride_with_gps():
    pass

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    return render_template('main.html')

@app.route('/rides')
def rides():
    encoder = gpolyencode.GPolyEncoder()
    a = DB.activities.find({'moving_time': 10123})[0]
    a = StravaActivity(a)

    df = a.make_df()
    
    return render_template(
        'poly.html',
        points = points,
        time=json.dumps(a.time.raw_data),
        distance=json.dumps(a.distance.raw_data),
        start=start)
if __name__ == '__main__':
    points = DB.activities.find_one()
    print points.keys()