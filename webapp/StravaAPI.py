import numpy as np
import pandas as pd
import requests as r
import pymongo
from datetime import date
import time
import calendar

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
        print response.json()

        return response.json()

    def store_activities(self):
        table = self.db.activities
        activities = self.list_activities()
        for activity in activities:
            if table.find_one({'id': activity['id']}) is not None:
                continue
            streams = self.get_stream(activity['id'])
            activity['streams'] = streams
            table.insert(activity)
            print 'Stored activity for %s on %s' % (
                                                    activity['athlete']['id'],
                                                    activity['start_date'])

    def get_stream(self, stream_id, types=None, stream_type='activity'):
        payload = {'resolution': 'medium'}
        if types is None:
            types = ['time','latlng','distance','altitude', 'moving',
                     'watts', 'velocity_smooth', 'moving', 'grade_smooth']

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
    strava = StravaAPI()
    # strava.store_efforts(7673423)
    # r = strava.list_activities()
    table = strava.db.activities
    table.remove()
    strava.store_activities()

    # for activity in table.find():
    #     # print activity['streams'].keys()
    #     for stream in activity['streams']:
    #         print stream, len(activity['streams'][stream]['data'])
    #         break






