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
        """
        INPUT: StravaDB
        OUTPUT: None

        Initialize a database object.
        """
        self.get_cursor()

    def get_cursor(self, local=True):
        """
        INPUT: StravaDB, BOOLEAN
        OUTPUT: None

        Get a connection and cursor object to the database.

        local indicates whether to use local or hosted database.
        """
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
        """
        INPUT: StravaDB, STRING, BOOLEAN
        OUTPUT: None

        Execute a query to the database.

        fetch specifies whether to get results or not.
        """
        try:
            self.cur.execute(query)
            if fetch:
                return self.cur.fetchall()
        except:
            print traceback.format_exc()
            print query
            self.conn.rollback()

    def create_athletes_table(self):
        """Create the athletes table"""

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
        """
        Create the activities table

        NOTES: the primary key id enforces uniqueness on the id
        column, however this should not be the case. (id, athlete_id)
        should be unique.
        """

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
                fitness10 REAL                    ,
                fitness30 REAL                    ,
                frequency10 INT(3)                ,
                frequency30 INT(3)                ,
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
        """Create the streams table"""

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
        """
        INPUT: StravaDB, 1D NUMPY ARRAY, 1D NUMPY ARRAY, 1D NUMPY ARRAY
        OUTPUT: 1D NUMPY ARRAY, 1D NUMPY ARRAY

        Get the moving time for this activities.

        Strava by default includes the non-moving time in the streams.
        These can be removed using the BOOLEAN stream 'moving', however,
        there are still times where the athlete is not moving but the 
        'moving' vector indicates otherwise. Use a heuristic to determine
        if they are moving and correct for that.
        """
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
        """
        INPUT: StravaDB, DICTIONARY, INT, INT
        OUTPUT: LIST

        Convert the raw streams from Strava to lists for DB insertion.
        """
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

    def create_routes_table(self):
        """Create the routes table"""

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
        """
        INPUT: StravaDB, GPX SEGMENT
        OUTPUT: DATAFRAME

        Build a streams dataframe from a GPX route.

        route is a gpxpy.segment
        """
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
        seg_frame['grade'] = np.append(np.diff(seg_frame['altitude']), 0) * 100 / \
                        np.append(np.diff(seg_frame['distance']), 0.01)
        return seg_frame.sort('distance')

    def create_route(self, gpx_file, athlete_id, name):
        """
        INPUT: StravaDB, STRING, INT, STRING
        OUTPUT: 1D NUMPY ARRAY, 1D NUMPY ARRAY

        Create a route from a gpx file.

        gpx_file is a string specifying the path to the .gpx file.
        athlete_id is the id of the athlete who this route belongs to.
        name is a string description of the route.
        """

        gpx_file = open(gpx_file, 'r')
        gpx = gpxpy.parse(gpx_file)
        route = gpx.tracks[0].segments[0]
        df = self.gpx_to_df(route)

        dt = datetime.now().date()

        cols = ['id', 'start_dt', 'distance', 'total_elevation_gain']

        q = """ SELECT %s
                FROM activities
                WHERE start_dt >= '%s'
                AND start_dt < '%s'
                AND athlete_id = %s
            """ % (', '.join(cols), dt - timedelta(30), dt, athlete_id)

        results = self.execute(q)
        difficulties10 = []
        difficulties30 = []
        for a in results:
            d = dict(zip(cols, a))
            print d
            dt2 = d['start_dt'].date()
            if dt2 < dt and dt2 >= dt - timedelta(30):
                difficulty = d['total_elevation_gain']*d['distance']
                difficulties30.append(difficulty)
            if dt2 < dt and dt2 >= dt - timedelta(10):
                difficulty = d['total_elevation_gain']*d['distance']
                difficulties10.append(difficulty)
        fitness10 = np.sum(difficulties10)
        fitness30 = np.sum(difficulties30)
        frequencies10 = len(difficulties10)
        frequencies30 = len(difficulties30)

        elevations = [point.elevation for point in route.points]
        ediff = np.diff(elevations)
        ediff = np.where(ediff > 1000, 0, ediff)
        climb = np.sum(ediff[np.where(ediff > 0)])


        d = {'start_dt': dt,
             'timezone': None,
             'city': None,
             'country': None,
             'start_longitude': route.points[0].longitude,
             'start_latitude': route.points[0].latitude,
             'distance': df.distance.iloc[-1],
             'fitness10': fitness10,
             'fitness10': fitness30,
             'frequency10': frequencies10,
             'frequency10': frequencies30,
             'name': name,
             'total_elevation_gain': climb,
             'athlete_count': 1,
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

    def insert_values(self, table_name, val_dict):
        """
        INPUT: StravaDB, STRING, DICTIONARY
        OUTPUT: BOOLEAN

        Insert an entry to a MySQL db tables.

        val_dict is a dictionary of (column, value) pairs.
        """

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
