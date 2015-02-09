import numpy as np
import pandas as pd
import requests as r
import pymongo

ACCESS_TOKEN = '535848c783c374e8f33549d22f089c1ce0d56cd6'

class StravaAPI(object):

    def __init__(self):
        self.base_url = 'https://www.strava.com/api/v3/'
        self.header = {'Authorization': 'Bearer %s' % ACCESS_TOKEN}
        self.access_token = ACCESS_TOKEN
        self.db = None
        self.client = pymongo.MongoClient()
        self.db = self.client.mydb

    def execute(self, url, payload={}):
        return r.get(self.base_url + url, headers=self.header, params=payload)

    def list_activities(self):
        url = 'athlete/activities'
        payload = {}
        response = self.execute(url, payload)
        
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
            types = ['time','latlng','distance','altitude',
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
        effort_ids = [(entry['effort_id'], entry['athlete_name']) for entry in data['entries']]

        efforts = []
        for eid in effort_ids:
            efforts.append(get_effort(eid[0], eid[1]))
        
        return efforts

    def get_effort(self, effort_id, name, types=None):
        if types is None:
            types = ['time','latlng','distance','altitude',
                     'watts', 'velocity_smooth', 'moving', 'grade_smooth']
        payload = {'resolution': 'medium'}
        url = 'segment_efforts/%s/streams/%s' % (effort_id, ','.join(types))
        response = self.execute(url, payload)
        data = response.json()
        data = {x['type']:x for x in data}
        data['name'] = name

        return data

    def store_efforts(self, segment_id):
        table = db.efforts
        efforts = get_efforts(segment_id)

        for effort in efforts:
            effort['segment_id'] = segment_id
            if table.find({'segment_id': segment_id, 'name': effort['name']}).count() != 0:
                print 'Duplicate! ', effort['name']
                continue
            table.insert(effort)


def main():
    segment_ids = [7673423, 4980024]


if __name__ == '__main__':
    strava = StravaAPI()
    # r = strava.list_activities()
    table = strava.db.activities
    # strava.store_activities()
    
    for activity in table.find():
        # print activity['streams'].keys()
        for stream in activity['streams']:
            print stream, len(activity['streams'][stream]['data'])
            break
    



    

