import numpy as np
import pymongo
from SignalProc import weighted_average, smooth, diff
from datetime import datetime
import matplotlib.pyplot as plt
from PlotTools import PlotTool
import seaborn as sns

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

class StravaEffort(object):

    def __init__(self, effort_dict):
        self.init(effort_dict)

    def init(self, effort_dict):
        self.id = effort_dict['id']
        self.name = effort_dict['name']
        self.athlete = effort_dict['athlete']#['id']
        self.date = self.strava_date(effort_dict['start_date'])
        stream_dict = effort_dict['streams']
        self.velocity = self.init_stream(stream_dict, 'velocity_smooth')
        self.distance = self.init_stream(stream_dict, 'distance')
        self.time = self.init_stream(stream_dict, 'time')
        self.grade = self.init_stream(stream_dict, 'grade_smooth')
        self.altitude = self.init_stream(stream_dict, 'altitude')

    def init_stream(self, stream_dict, stream_type):
        return StravaStream(stream_type, stream_dict[stream_type])

    def stream_types(self):
        return {'velocity_smooth', 'distance', 'time', 'grade_smooth', 'altitude'}

    def strava_date(self, date_string):
        if type(date_string) != unicode:
            return date_string
        return datetime.strptime(date_string, '%Y-%m-%dT%H:%M:%SZ')

class StravaActivity(StravaEffort):

    def __init__(self, activity_dict):
        self.init(activity_dict)
        self.distance.convert_units()
        self.velocity.convert_units()
        self.altitude.convert_units()
        self.city = activity_dict['location_city']
        self.total_distance = activity_dict['distance']
        self.hills = self.hill_analysis()
        self.is_valid_ride = self.is_ride()

    def init_stream(self, stream_dict, stream_type):
        return StravaStream(stream_type, stream_dict[stream_type])

    def is_ride(self):
        min_vel = 8.  # mph
        max_vel = 30.  # mph

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
                    stream_dict = self.make_stream_dict()
                    for key in stream_dict:
                        stream_dict[key]['data'] = stream_dict[key]['data'][hill_start_idx:k]

                    effort_dict = {'id': len(hills),
                                   'name': self.name,
                                   'athlete': self.athlete,
                                   'start_date': self.date,
                                   'streams': stream_dict}
                    hill = StravaHill(effort_dict, self, previous_hill=prev_hill)
                    hills.append(hill)

        return self.clean_hills(hills)

    def make_stream_dict(self):
        return {stream_type: {'data': getattr(self, stream_type.replace('_smooth', '')).raw_data} \
                for stream_type in self.stream_types()}

    def clean_hills(self, hills):
        min_score = 8000. / meters_per_mile
        min_separation = 0.5  # miles
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
        # stream_dict = self.make_stream_dict()
        # for key in stream_dict:
        #     stream_dict[key]['data'] = stream_dict[key]['data'][hill_start_idx:k]
        stream_dict = {}
        for stream_type in self.stream_types():
            hill1_data = getattr(hill1, stream_type.replace('_smooth','')).raw_data
            hill2_data = getattr(hill2, stream_type.replace('_smooth','')).raw_data
            stream_dict[stream_type] = {}
            stream_dict[stream_type]['data'] = np.append(hill1_data, hill2_data)
        effort_dict = {'id': hill1.id,
                       'name': hill1.name,
                       'athlete': hill1.athlete,
                       'start_date': hill1.date,
                       'streams': stream_dict}
        hill = StravaHill(effort_dict, self, previous_hill=hill1.previous_hill)
        # hill = StravaHill(hill1.activity,
        #                   hill1.start,
        #                   hill2.stop,
        #                   hill1.previous_climb,
        #                   hill1.previous_hill)
        return hill

    def plot_hills(self):
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        p = PlotTool()
        ax.plot(self.distance.raw_data, self.altitude.raw_data,
                c='r', label='altitude')
        xmin, xmax = np.min(self.distance.raw_data), np.max(self.distance.raw_data)
        ymin, ymax = np.min(self.altitude.raw_data), np.max(self.altitude.raw_data)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_title(self.date.date())
        axis_label_font = 20
        ax.set_xlabel('Distance (miles)', fontsize=axis_label_font)
        ax.set_ylabel('Altitude (ft)', fontsize=axis_label_font)
        ax.set_title('%s, %s, %s' % \
                        (self.name, self.athlete['id'], self.date.date()),
                        fontsize=24)

        ax.tick_params(labelsize=16)

        cmap = plt.get_cmap("YlOrRd")

        for hill in self.hills:
            label = 'Score: %0.0f' % (hill.score)
            X = hill.distance.raw_data
            Y = hill.altitude.raw_data
            p.plot_fill(X, Y, hill.velocity.filtered, cmap, ax)

        plt.show()


    def __repr__(self):
        return '<%s, %s, %s, %s>' % \
            (self.name, self.date, self.city, len(self.distance.raw_data))


class StravaStream(object):

    def __init__(self, stream_type, stream_dict, change_units=True):
        self.raw_data = np.array(stream_dict['data'])
        self.stream_type = stream_type
        self.filter()
    
    def filter(self):
        self.filtered = smooth(self.raw_data)

    def convert_units(self):
        if self.stream_type == 'distance':
            self.raw_data /= meters_per_mile
        elif self.stream_type == 'velocity':
            self.raw_data *= 3600
            self.raw_data /= meters_per_mile
        elif self.stream_type == 'altitude':
            self.raw_data *= feet_per_meter


class StravaHill(StravaEffort):

    def __init__(self, effort_dict, activity, previous_hill=None):
        self.init(effort_dict)
        self.activity = activity
        self.previous_hill = previous_hill
        self.score = self.hill_score()

    def hill_score(self):
        grade_vec = self.grade.raw_data
        dist_vec = self.distance.raw_data

        mean_grade = weighted_average(grade_vec, dist_vec)
        distance = self.distance.raw_data[-1] - self.distance.raw_data[0]

        return distance*mean_grade

    def __repr__(self):
        return '<%s, %s>' % \
            (self.score, len(self.distance.raw_data))

if __name__ == '__main__':
    u = StravaUser('Seth')
    u.activities[3].plot_hills()
    # for hill in u.activities[2].hills:
    #     print hill.score, 8000 / meters_per_mile
