import numpy as np
import pymongo
from SignalProc import weighted_average, smooth, diff
from datetime import datetime

# global vars
meters_per_mile = 1609.34
feet_per_meter = 3.280


class StravaUser(object):

    def __init__(self, name):
        self.activities = None
        self.name = name

        self.get_activities()

    def get_activities(self, query={}, min_length=990):
        client = pymongo.MongoClient()
        db = client.mydb
        table = db.activities
        
        result = table.find(query)
        activities = [activity for activity in result]

        self.activities = []
        for activity in activities:
            a = StravaActivity(activity)
            if a.is_ride() and len(a.distance.raw_data) > min_length:
                self.activities.append(a)


class StravaActivity(object):

    def __init__(self, activity_dict):
        self.id = activity_dict['id']
        self.name = activity_dict['name']
        self.athlete = activity_dict['athlete']
        self.city = activity_dict['location_city']
        self.date = self.strava_date(activity_dict['start_date'])
        self.total_distance = activity_dict['distance']
        stream_dict = activity_dict['streams']
        self.velocity = self.init_stream(stream_dict, 'velocity_smooth')
        self.distance = self.init_stream(stream_dict, 'distance')
        self.time = self.init_stream(stream_dict, 'time')
        self.grade = self.init_stream(stream_dict, 'grade_smooth')
        self.altitude = self.init_stream(stream_dict, 'altitude')
        self.hills = self.hill_analysis()
        self.is_valid_ride = self.is_ride()

    def init_stream(self, stream_dict, stream_type):
        return StravaStream(stream_type, stream_dict[stream_type])

    def strava_date(self, date_string):
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

    def is_ride(self):
        min_vel = 8. # mph
        max_vel = 30. # mph

        avg_vel = np.mean(self.velocity.raw_data)

        return avg_vel >= min_vel and avg_vel <= max_vel

    def hill_analysis(self):
        diff_alt = diff(self.altitude.raw_data[:, np.newaxis],
                        self.distance.raw_data[:, np.newaxis])
        diff_alt = smooth(diff_alt)

        hills = []
        hill_start = False
        hill_start_idx = 0
        hill_start_time = 0
        stream_length = self.distance.raw_data.shape[0]
        climb_total = np.zeros(stream_length)
        for k in xrange(stream_length):
            time_elapsed = self.time.raw_data[k] - hill_start_time
            if k != 0:
                climb = np.max([self.altitude.raw_data[k] - self.altitude.raw_data[k - 1], 0])
                climb_total[k] = climb_total[k - 1] + climb

            if diff_alt[k] > 0 and not hill_start:
                hill_start = True
                hill_start_idx = k
                hill_start_time = self.time.raw_data[hill_start_idx]

            elif hill_start and (diff_alt[k] < 0 or k == stream_length - 1):
                hill_start = False
                if len(hills) != 0:
                    prev_hill = hills[-1]
                else:
                    prev_hill = None
                if k - hill_start_idx >= 2:
                    hill = StravaHill(self, hill_start_idx, k, climb_total, prev_hill)
                    hills.append(hill)

        return self.clean_hills(hills)  

    def clean_hills(self, hills):
        min_score = 8000. / meters_per_mile
        min_separation = 0.5 # miles
        clean_hills = []
        for k, hill in enumerate(hills):
            if len(clean_hills) != 0:
                if hill.distance.raw_data[0] - clean_hills[-1].distance.raw_data[-1] < min_separation:
                    hill = self.merge_hills(clean_hills[-1], hill)
                    clean_hills.pop()
            if hill.score > min_score:
                clean_hills.append(hill)

        return clean_hills

    def merge_hills(self, hill1, hill2):
        hill = StravaHill(hill1.activity,
                          hill1.start,
                          hill2.stop, 
                          hill1.previous_climb,
                          hill1.previous_hill)
        return hill    

    def __repr__(self):
        return '<%s, %s, %s, %s>' % \
            (self.name, self.date, self.city, len(self.distance.raw_data))

class StravaStream(object):

    def __init__(self, stream_type, stream_dict, change_units=True):
        self.raw_data = np.array(stream_dict['data'])
        if stream_type == 'distance' and change_units:
            self.raw_data /= meters_per_mile
        elif stream_type == 'velocity' and change_units:
            self.raw_data *= 3600
            self.raw_data /= meters_per_mile
        elif stream_type == 'altitude' and change_units:
            self.raw_data *= feet_per_meter
        self.stream_type = stream_type
        self.filtered = smooth(self.raw_data)

class StravaHill(object):

    def __init__(self, activity, start, stop, climb_total, previous_hill=None):
        self.activity = activity

        self.velocity = self.init_stream('velocity', self.activity.velocity.raw_data[start:stop])
        self.distance = self.init_stream('distance', self.activity.distance.raw_data[start:stop])
        self.time = self.init_stream('time', self.activity.time.raw_data[start:stop])
        self.grade = self.init_stream('grade', self.activity.grade.raw_data[start:stop])
        self.altitude = self.init_stream('altitude', self.activity.altitude.raw_data[start:stop])
        
        self.previous_distance = self.activity.distance.raw_data[start]
        self.previous_climb = climb_total
        self.start = start
        self.stop = stop
        self.time_riding = self.activity.time.raw_data[start] - self.activity.time.raw_data[0]
        self.previous_hill = previous_hill
        self.score = self.hill_score()
        # print self.score

    def init_stream(self, stream_type, data):
        return StravaStream(stream_type, {'data': data}, change_units=False)

    def hill_score(self):
        grade_vec = self.grade.raw_data
        dist_vec = self.distance.raw_data

        mean_grade = weighted_average(grade_vec, dist_vec)
        distance = self.distance.raw_data[-1] - \
                   self.distance.raw_data[0] 

        return distance*mean_grade
    
    def __repr__(self):
        return '<%s, %s>' % \
            (self.score, len(self.distance.raw_data))

if __name__ == '__main__':
    u = StravaUser('Seth')
    print u.activities[2].hills[0]
