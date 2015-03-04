import MySQLdb as sql
import pymongo
import traceback
from datetime import datetime, date, timedelta
import numpy as np
import gpxpy
import pandas as pd
import collections

CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
MongoDB = CLIENT.strava

class StravaDB(object):

    def __init__(self):
        self.get_cursor()

    def get_cursor(self, local=True):
        if local:
            self.conn = sql.connect(host="127.0.0.1",
                              user="root",
                              passwd="abc123",
                              db="hendris$strava")
        else:
            self.conn = sql.connect(host="127.0.0.1",
                                  user="hendris",
                                  passwd="abc123",
                                  db="hendris$strava")
        self.conn.autocommit(False)
        self.cur = self.conn.cursor()

    def execute(self, query, fetch=True):
        try:
            self.cur.execute(query)
            if fetch:
                return self.cur.fetchall()
        except:
            print traceback.format_exc()
            print query
            self.conn.rollback()

    def show_tables(self):
        q = """ SHOW tables;
            """

        print self.execute(q)

    def create_athletes_table(self):
        q = """ CREATE TABLE athletes
                (
                id INT    PRIMARY KEY    NOT NULL,
                firstname CHAR(20)       NOT NULL,
                lastname CHAR(20)        NOT NULL,
                sex CHAR(1)              NOT NULL,
                city CHAR(30)            NOT NULL,
                state CHAR(10)           NOT NULL,
                country CHAR(40)         NOT NULL,
                access_key CHAR(100)     NOT NULL,
                UNIQUE(id)
                );
            """
        self.execute(q, fetch=False)

    def create_activities_table(self):
        q = """ CREATE TABLE activities
                (
                id INT PRIMARY KEY        NOT NULL,
                start_dt TIMESTAMP        NOT NULL,
                timezone CHAR(40)                 ,
                city CHAR(40)                     ,
                country CHAR(40)                  ,
                start_longitude REAL      NOT NULL,
                start_latitude REAL       NOT NULL,
                elapsed_time INT                  ,
                distance REAL             NOT NULL,
                moving_time INT           NOT NULL,
                fitness_level REAL                ,
                average_speed REAL                ,
                kilojoules REAL                   ,
                max_speed REAL                    ,
                name CHAR (100)                   ,
                total_elevation_gain REAL         ,
                athlete_id INT NOT NULL REFERENCES athletes(id),
                UNIQUE(id, athlete_id)
                );
            """
        self.execute(q, fetch=False)

    def create_streams_table(self):
        q = """ CREATE TABLE streams
                (
                activity_id INT NOT NULL REFERENCES activities(id),
                athlete_id INT NOT NULL REFERENCES athletes(id),
                time REAL                 NOT NULL,
                distance REAL             NOT NULL,
                grade REAL                NOT NULL,
                altitude REAL             NOT NULL,
                velocity REAL             NOT NULL,
                latitude REAL             NOT NULL,
                longitude REAL            NOT NULL,
                moving BOOLEAN            NOT NULL,
                UNIQUE(activity_id, athlete_id, time)
                );
            """
        self.execute(q, fetch=False)

    def get_moving(self, moving, distance, time):
        not_moving = np.where(~moving)[0]
        for ind in not_moving:
            time[ind:] -= (time[ind] - \
                                         time[ind -1])

        dd = np.diff(distance)
        dt = np.diff(time)
        not_moving = np.where(dd/dt < 1)[0]
        for ind in not_moving:
            ind += 1
            time[ind:] -= (time[ind] - time[ind -1])

        return time, distance

    def process_streams(self, stream_dict, athlete_id, activity_id):
        # stream_dict = activity['streams']
        distance = np.array(stream_dict['distance']['data'])
        time = np.array(stream_dict['time']['data'])
        velocity = np.array(stream_dict['velocity_smooth']['data'])
        grade = np.array(stream_dict['grade_smooth']['data'])
        altitude = np.array(stream_dict['altitude']['data'])
        latlng = np.array(stream_dict['latlng']['data'])
        moving = np.array(stream_dict['moving']['data'])
        latitude = latlng[:,0]
        longitude = latlng[:,1]
        time, distance = self.get_moving(moving, distance, time)

        new_time = np.linspace(time[0], time[-1], time.shape[0])

        latitude = np.interp(new_time, time, latitude)
        longitude = np.interp(new_time, time, longitude)
        distance = np.interp(new_time, time, distance)
        velocity = np.interp(new_time, time, velocity)
        grade = np.interp(new_time, time, grade)
        altitude = np.interp(new_time, time, altitude)
        athlete_ids = np.array([athlete_id]*new_time.shape[0])
        activity_ids = np.array([activity_id]*new_time.shape[0])

        zipped = zip(activity_ids, athlete_ids, new_time, distance, grade, altitude, velocity, latitude, longitude)
        return [list(x) for x in zipped]


    def move_streams(self):
        find = {'id': 1, 'athlete': 1, 'streams': 1, 'start_date_local': 1}
        activities = MongoDB.activities.find({}, find)
        for a in activities:
            # a = MongoDB.activities.find_one()
            start_time = datetime.strptime(a['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
            data = self.process_streams(a)
            print a['start_date_local'], len(data)

            q = """ INSERT INTO streams (activity_id, athlete_id, time, distance, grade, altitude, velocity, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            self.cur.executemany(q, data)
        self.conn.commit()

    def create_routes_table(self):
        q = """ CREATE TABLE routes
                (
                id SERIAL PRIMARY KEY     NOT NULL,
                start_dt TIMESTAMP        NOT NULL,
                timezone CHAR(40)                 ,
                city CHAR(40)                     ,
                country CHAR(40)                  ,
                start_longitude REAL      NOT NULL,
                start_latitude REAL       NOT NULL,
                distance REAL             NOT NULL,
                fitness_level REAL                ,
                name CHAR (100)                   ,
                total_elevation_gain REAL         ,
                athlete_id INT NOT NULL REFERENCES athletes(id)
                );
            """
        self.execute(q, fetch=False)

    def gpx_to_df(self, route):
        d = collections.defaultdict(list)

        previous_point = None
        for point in route.points:
            d['latitude'].append(point.latitude)
            d['longitude'].append(point.longitude)

            if point.elevation < 0:
                d['altitude'].append(previous_point.elevation)
            else:
                d['altitude'].append(point.elevation)
            if previous_point is not None:
                d['distance'].append(point.distance_2d(previous_point))
            else:
                d['distance'].append(0)

            previous_point = point

        d['distance'] = np.cumsum(d['distance'])
        new_dist = np.linspace(d['distance'][0], d['distance'][-1], 9000)
        d['altitude'] = np.interp(new_dist, d['distance'], d['altitude'])
        d['latitude'] = np.interp(new_dist, d['distance'], d['latitude'])
        d['longitude'] = np.interp(new_dist, d['distance'], d['longitude'])
        d['distance'] = new_dist

        seg_frame = pd.DataFrame(d)
        # seg_frame['distance'] = np.cumsum(seg_frame['distance'])
        seg_frame['grade'] = np.append(np.diff(seg_frame['altitude']), 0) * 100 / \
                        np.append(np.diff(seg_frame['distance']), 0.01)
        return seg_frame.sort('distance')

    def create_route(self, gpx_file, athlete_id, name):
        gpx_file = open(gpx_file, 'r')
        gpx = gpxpy.parse(gpx_file)
        route = gpx.tracks[0].segments[0]
        df = self.gpx_to_df(route)

        d = {'start_dt': datetime.now(),
             'timezone': None,
             'city': None,
             'country': None,
             'start_longitude': route.points[0].longitude,
             'start_latitude': route.points[0].latitude,
             'distance': df.distance.iloc[-1],
             'fitness_level': 0,
             'name': name,
             'total_elevation_gain': 0,
             'athlete_id': athlete_id
            }

        self.insert_values('routes', d)
        self.cur.execute("""SELECT LAST_INSERT_ID();""")
        activity_id = self.cur.fetchone()[0]

        df['velocity'] = [-9999]*df.shape[0]
        df['moving'] = [True]*df.shape[0]
        df['time'] = np.arange(-1, -df.shape[0] - 1, -1)
        df['athlete_id'] = [athlete_id]*df.shape[0]
        df['activity_id'] = [activity_id]*df.shape[0]
        zipped = zip(df.activity_id.values,
                     df.athlete_id.values,
                     df.time.values,
                     df.distance.values,
                     df.grade.values,
                     df.altitude.values,
                     df.velocity.values,
                     df.latitude.values,
                     df.longitude.values)
        data = [list(x) for x in zipped]
        print data[:5]
        q = """ INSERT INTO streams (activity_id, athlete_id, time, distance, grade, altitude, velocity, latitude, longitude)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
        self.cur.executemany(q, data)
        self.conn.commit()


    def move_activities(self):
        find = {'id': 1, 'start_date_local': 1, 'timezone': 1,
                    'location_city': 1, 'location_country': 1,
                    'start_longitude': 1, 'start_latitude': 1,
                    'elapsed_time': 1, 'distance': 1, 'moving_time': 1,
                    'fitness_level': 1, 'average_speed': 1, 'kilojoules': 1,
                    'max_speed': 1, 'name': 1, 'total_elevation_gain': 1,
                    'athlete': 1}
        activities = MongoDB.activities.find({}, find)
        for a in activities:
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
                 'fitness_level': 0,
                 'average_speed': a['average_speed'],
                 'max_speed': a['max_speed'],
                 'name': a['name'],
                 'total_elevation_gain': a['total_elevation_gain'],
                 'athlete_id': a['athlete']['id']
            }
            self.insert_values('activities', d)

    def move_athletes(self):
        athletes = MongoDB.athletes.find()
        for ath in athletes:
            d = {'id': ath['id'],
                 'firstname': ath['firstname'],
                 'lastname': ath['lastname'],
                 'sex': ath['sex'],
                 'city': ath['city'],
                 'state': ath['state'],
                 'country': ath['country']}
            self.insert_values('athletes', d)

    def insert_values(self, table_name, val_dict):

        keys = val_dict.keys()
        values = [val_dict[k] for k in keys]
        q = """ INSERT INTO {table} ({keys})
                VALUES ({placeholders})
            """.format(
                      table = table_name,
                      keys = ', '.join(keys),
                      placeholders = ', '.join([ "%s" for v in values ])
                      )
        try:
            self.cur.execute(q, values)
            self.conn.commit()
            return True
        except:
            print traceback.format_exc()
            print q
            return False

if __name__ == '__main__':
    main()

    """ CREATE TABLE fitness AS
        
    """


    """ UPDATE 
            activities a, 
            (SELECT 
                a_dt as dt,
                id as activity_id,
                athlete_id as athlete_id,
                SUM(score) as fitness_score
            FROM (SELECT
                    a.id,
                    a.athlete_id,
                    a.start_dt AS a_dt,
                    b.total_elevation_gain,
                    b.distance,
                    b.distance*b.total_elevation_gain AS score,
                    b.start_dt
                FROM activities a
                JOIN activities b
                ON b.start_dt >= DATE_SUB(a.start_dt, INTERVAL 30 DAY)
                AND b.start_dt < a.start_dt
                AND a.athlete_id = b.athlete_id
                ORDER BY a_dt) scores
            GROUP BY a_dt, id, athlete_id) f
        SET a.fitness_level = f.fitness_score
        WHERE a.id = f.activity_id
        AND a.athlete_id = f.athlete_id;
    """