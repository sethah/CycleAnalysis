import numpy as np
import pandas as pd
import requests as r
import pymongo
from datetime import date, timedelta, datetime
import time
import calendar
from StravaDB import StravaDB
import traceback
import MySQLdb

ACCESS_TOKEN = '535848c783c374e8f33549d22f089c1ce0d56cd6'

class StravaAPI(object):

    def __init__(self, access_token=ACCESS_TOKEN):
        """
        INPUT: StravaAPI, STRING
        OUTPUT: None

        Initialize a StravaAPI object.

        access_token is the access token for the user who's data will be retrieved.

        This class is used to interface with the Strava API. A client id and
        access token are required. These are provided by Strava after registering
        for a developer account.
        """
        self.access_token = access_token
        self.client_secret = '4c72c397da5890b14a7f71c02d9f5a58c24ceec5'
        self.client_id = 4554
        self.base_url = 'https://www.strava.com/api/v3/'
        self.header = {'Authorization': 'Bearer %s' % self.access_token}
        self.client = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
        self.db = self.client.strava

    def execute(self, url, payload={}):
        """
        INPUT: StravaAPI, STRING, DICTIONARY
        OUTPUT: REQUESTS.RESPONSE

        Query the Strava API and return the response.

        url is the url suffix for the query. Different data types have
        differt suffixes.
        payload is a dictionary of query arguments for the query.
        """
        return r.get(self.base_url + url, headers=self.header, params=payload)

    def exchange_token(self, code):
        """
        INPUT: StravaAPI, STRING
        OUTPUT: JSON

        Retrieve an access key from the Strava API. 

        code is a string provided by the Strava authentication page and can be
        exchanged for an access key for the authorized user.
        """
        payload = {'client_id': self.client_id, 'client_secret': self.client_secret, 'code': code}
        response = r.post('https://www.strava.com/oauth/token', headers=self.header, params=payload)

        return response.json()

    def list_activities(self, start_dt=None):
        """
        INPUT: StravaAPI, STRING
        OUTPUT: JSON

        List the activities for a given user between now and start_dt.

        start_dt is a string specifying the lower time bound.
        """
        url = 'athlete/activities'
        if start_dt is not None:
            after = calendar.timegm(time.strptime(start_dt, '%Y-%m-%d'))
        else:
            after = calendar.timegm(time.strptime('2014-01-01', '%Y-%m-%d'))
        before = calendar.timegm(time.gmtime())
        payload = {'before': before, 'after': after, 'per_page': 100}
        response = self.execute(url, payload)

        return response.json()

    def store_activities(self, start_dt=None, store_streams=True, max_activities=50):
        """
        INPUT: StravaAPI, STRING, BOOLEAN
        OUTPUT: None

        Retrieve and store the activities for a user.

        start_dt is a string specifying the lower time bound for activities.
        store_streams is a Boolean which indicates whether or not to store
        the raw data streams for the activitiy.
        """
        DB = StravaDB()
        table = self.db.activities
        activities = self.list_activities(start_dt=start_dt)
        activities = self.fitness_score(activities)
        for i, a in enumerate(activities):
            if i >= max_activities:
                break
            if a['type'] != 'Ride':
                continue
            try:
                self.store_activity(a, store_streams=store_streams)
            except:
                print 'failed storing activity %s' % a['start_date_local']
                continue
        DB.conn.commit()

    def store_activity(self, activity, store_streams=False):
        """
        INPUT: StravaAPI, JSON, BOOLEAN
        OUTPUT: None

        Store a single activity in the database.

        NOTES: This should be in the StravaDB object, not here.
        """
        DB = StravaDB()
        a = activity

        d = {'id': a['id'],
             'start_dt': datetime.strptime(a['start_date_local'], '%Y-%m-%dT%H:%M:%SZ'),
             'timezone': a['timezone'],
             'city': a['location_city'],
             'country': a['location_country'],
             'start_longitude': a['start_longitude'],
             'start_latitude': a['start_latitude'],
             'elapsed_time': a['elapsed_time'],
             'distance': a['distance'],
             'moving_time': a['moving_time'],
             'fitness10': a.get('fitness10', 0),
             'fitness30':  a.get('fitness30', 0),
             'frequency10':  a.get('frequencies10', 0),
             'frequency30':  a.get('frequencies30', 0),
             'average_speed': a['average_speed'],
             'max_speed': a['max_speed'],
             'name': a['name'],
             'total_elevation_gain': a['total_elevation_gain'],
             'athlete_id': a['athlete']['id']
        }
        if store_streams:
            data = DB.process_streams(self.get_stream(a['id']), a['athlete']['id'], a['id'])
            print len(data)

            try:
                q = """ INSERT INTO streams (activity_id, athlete_id, time, distance, grade, altitude, velocity, latitude, longitude)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """
                DB.cur.executemany(q, data)
            except MySQLdb.IntegrityError:
                print traceback.format_exc()
        try:
            DB.insert_values('activities', d)
        except MySQLdb.IntegrityError:
            print traceback.format_exc()
        

        print 'Stored activity for %s on %s' % (
                                                a['athlete']['id'],
                                                a['start_date'])

    def fitness_score(self, activities):
        """
        INPUT: StravaAPI, LIST
        OUTPUT: LIST

        Compute fitness levels for each activity.

        activities is a list containing JSON descriptions of the activities.
        """
        for i, a1 in enumerate(activities):
            score = 0
            difficulties10 = []
            difficulties30 = []
            dt = datetime.strptime(a1['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
            for a2 in activities[i:]:
                dt2 = datetime.strptime(a2['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
                if dt2 < dt and dt2 >= dt - timedelta(30):
                    difficulty = a2['total_elevation_gain']*a2['distance']
                    difficulties30.append(difficulty)
                if dt2 < dt and dt2 >= dt - timedelta(10):
                    difficulty = a2['total_elevation_gain']*a2['distance']
                    difficulties10.append(difficulty)

            a1['fitness10'] = np.sum(difficulties10)
            a1['fitness30'] = np.sum(difficulties30)
            a1['frequencies10'] = len(difficulties10)
            a1['frequencies30'] = len(difficulties30)

        return activities

    def get_activity(self, activity_id):
        """
        INPUT: StravaAPI, INT
        OUTPUT: JSON

        Get a single activity from the Strava API.

        activity_id is the integer id assigned to the activity by Strava.
        """
        url = 'activities/%s' % activity_id
        payload = {}
        response = self.execute(url, payload)

        return response.json()

    def get_stream(self, stream_id, types=None, stream_type='activity'):
        payload = {'resolution': 'high'}
        if types is None:
            types = ['time','latlng','distance','altitude', 'moving',
                     'velocity_smooth', 'moving', 'grade_smooth']

        if stream_type == 'activity':
            url = 'activities/%s/streams/%s' % (stream_id, ','.join(types))
        else:
            url = ''

        response = self.execute(url, payload)
        try:
            data = response.json()
            data = {x['type']:x for x in data}
        except:
            print response.json()
            raise

        return data

    def get_efforts(self, segment_id):
        url = 'segments/%s/leaderboard' % segment_id
        payload = {'per_page': 10}
        response = self.execute(url, payload)
        data = response.json()

        efforts = []
        for effort in data['entries']:
            streams = self.get_effort_streams(effort['effort_id'])
            effort['streams'] = streams
            efforts.append(effort)

        return efforts

    def get_effort_streams(self, effort_id, types=None):
        if types is None:
            types = ['time','latlng','distance','altitude',
                     'watts', 'velocity_smooth', 'moving', 'grade_smooth']
        payload = {'resolution': 'medium'}
        url = 'segment_efforts/%s/streams/%s' % (effort_id, ','.join(types))
        response = self.execute(url, payload)
        data = response.json()
        data = {x['type']:x for x in data}

        return data

    def store_efforts(self, segment_id):
        table = self.db.efforts
        efforts = self.get_efforts(segment_id)

        for effort in efforts:
            effort['segment_id'] = segment_id
            table.insert(effort)
