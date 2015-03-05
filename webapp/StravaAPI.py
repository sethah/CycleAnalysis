import numpy as np
import pandas as pd
import requests as r
import pymongo
from datetime import date, timedelta, datetime
import time
import calendar
from StravaDB import StravaDB
import traceback

ACCESS_TOKEN = '535848c783c374e8f33549d22f089c1ce0d56cd6'
# ACCESS_TOKEN = 'c12c290c0e9241b09314c850cd24ce97e036ac4f'


class StravaAPI(object):

    def __init__(self, access_token=ACCESS_TOKEN):
        self.access_token = access_token
        self.client_secret = '4c72c397da5890b14a7f71c02d9f5a58c24ceec5'
        self.client_id = 4554
        self.base_url = 'https://www.strava.com/api/v3/'
        self.header = {'Authorization': 'Bearer %s' % self.access_token}
        self.client = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
        self.db = self.client.strava

    def execute(self, url, payload={}):
        # print self.base_url + url
        return r.get(self.base_url + url, headers=self.header, params=payload)

    def exchange_token(self, code):
        payload = {'client_id': self.client_id, 'client_secret': self.client_secret, 'code': code}
        response = r.post('https://www.strava.com/oauth/token', headers=self.header, params=payload)

        return response.json()

    def list_activities(self):
        url = 'athlete/activities'
        after = calendar.timegm(time.strptime('2014-01-01', '%Y-%m-%d'))
        before = calendar.timegm(time.gmtime())
        payload = {'before': before, 'after': after, 'per_page': 100}
        response = self.execute(url, payload)

        return response.json()

    def recent_activities(self, athlete_id):
        table = self.db.activities
        find = {'id': 1, 'total_elevation_gain': 1, 'distance': 1, 'start_date_local': 1}
        activities = list(table.find({'athlete.id': athlete_id}, find))
        print len(activities)
        for a in activities:
            dt1 = datetime.strptime(a['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
            levels = []
            for x in activities:
                dt2 = datetime.strptime(x['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
                if ((dt2 >= dt1 - timedelta(30)) and (dt2 < dt1)):
                    levels.append(x['distance']*x['total_elevation_gain'])
            level = np.sum(levels)
            level = np.min([1e9, level])
            level /= 1e9
            table.update({'athlete.id': athlete_id, 'id': a['id']},
                         {'$set': {'fitness_level': level}})

    def store_activities(self, store_streams=True):
        DB = StravaDB()
        table = self.db.activities
        activities = self.list_activities()
        activities = self.fitness_score(activities)
        for a in activities:
            if a['type'] != 'Ride':
                continue

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
                 'fitness10': a['fitness10'],
                 'fitness30': a['fitness30'],
                 'frequency10': a['frequencies10'],
                 'frequency30': a['frequencies30'],
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
                except IntegrityError:
                    print traceback.format_exc()
                    continue
            # break
            try:
                DB.insert_values('activities', d)
            except IntegrityError:
                print traceback.format_exc()
                continue

            

            print 'Stored activity for %s on %s' % (
                                                    a['athlete']['id'],
                                                    a['start_date'])
        DB.conn.commit()

    def fitness_score(self, activities):
        # rides in last 10 days * avg difficult of ride
        # rides in last 30 days * avg difficulty of ride
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

    def downsample(self, t, vec, num_samples=5000):
        if len(vec) < num_samples:
            return vec
        new_t = np.linspace(t[0], t[-1], num_samples)
        print len(new_t), len(t), len(vec)
        return np.interp(new_t, np.array(t), np.array(vec)).tolist()

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
        data = response.json()
        data = {x['type']:x for x in data}

        return data

    def get_efforts(self, segment_id):
        url = 'segments/%s/leaderboard' % segment_id
        payload = {'per_page': 10}
        response = self.execute(url, payload)
        data = response.json()
        # effort_ids = [(entry['effort_id'], entry['athlete_name']) for entry in data['entries']]
        print data['entries'][0].keys()
        efforts = []
        for effort in data['entries']:
            # efforts.append(get_effort(eid[0], eid[1]))
            # effort = get_effort
            streams = self.get_effort_streams(effort['effort_id'])
            # print streams
            effort['streams'] = streams
            efforts.append(effort)
            # break

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
        # data['name'] = name

        return data

    def store_efforts(self, segment_id):
        table = self.db.efforts
        # table.remove()
        efforts = self.get_efforts(segment_id)

        for effort in efforts:
            effort['segment_id'] = segment_id
            table.insert(effort)

def main():
    segment_ids = [7673423, 4980024]



if __name__ == '__main__':
    pass
    # strava = StravaAPI()
    # # strava.store_efforts(7673423)
    # # r = strava.list_activities()
    # table = strava.db.activities
    # table.remove()
    # strava.store_activities()

    # for activity in table.find():
    #     # print activity['streams'].keys()
    #     for stream in activity['streams']:
    #         print stream, len(activity['streams'][stream]['data'])
    #         break





