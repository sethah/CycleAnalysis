import MySQLdb as sql
import pymongo
import traceback
from datetime import datetime, date, timedelta
import numpy as np

CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
MongoDB = CLIENT.strava

class StravaDB(object):

    def __init__(self):
        self.get_cursor()

    def get_cursor(self):
        self.db = sql.connect(host="127.0.0.1",
                              user="hendris",
                              passwd="abc123",
                              db="hendris$strava")
        self.db.autocommit(False)
        self.cur = self.db.cursor()

    def execute(self, query, fetch=True):
        try:
            self.cur.execute(query)
            if fetch:
                return self.cur.fetchall()
        except:
            print query
            self.db.rollback()

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

    def process_streams(self, activity):
        stream_dict = activity['streams']
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
        athlete_ids = np.array([activity['athlete']['id']]*new_time.shape[0])
        activity_ids = np.array([activity['id']]*new_time.shape[0])

        # start_time = datetime.strptime(activity['start_date_local'], '%Y-%m-%dT%H:%M:%SZ')
        # deltas = np.array(map(lambda x: timedelta(seconds=x), new_time))
        # tmstmp = deltas + start_time

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


        # for i in xrange(len(a['streams']['time']['data'])):
        #     if i > 10:
        #         break
        #     data[str(i)] = [a['id'], a['athlete']['id'],
        #               start_time + timedelta(seconds=a['streams']['time']['data'][i]),
        #               a['streams']['distance']['data'][i],
        #               a['streams']['grade_smooth']['data'][i],
        #               a['streams']['altitude']['data'][i],
        #               a['streams']['velocity_smooth']['data'][i],
        #               a['streams']['latlng']['data'][i][0],
        #               a['streams']['latlng']['data'][i][1]
        #               ]
            # d = {'activity_id': a['id'],
            #      'athlete_id': a['athlete']['id'],
            #      'tmstmp': start_time + timedelta(seconds=a['streams']['time']['data'][i]),
            #      'distance': a['streams']['distance']['data'][i],
            #      'altitude': a['streams']['altitude']['data'][i],
            #      'velocity': a['streams']['velocity_smooth']['data'][i],
            #      'latitude': a['streams']['latlng']['data'][i][0],
            #      'longitude': a['streams']['latlng']['data'][i][1],
            #      'grade': a['streams']['grade_smooth']['data'][i]}
        # print data
        # break
            q = """ INSERT INTO streams (activity_id, athlete_id, time, distance, grade, altitude, velocity, latitude, longitude)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            self.cur.executemany(q, data)
        self.db.commit()
        # break

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
            self.db.commit()
            return True
        except:
            print traceback.format_exc()
            print q
            return False

if __name__ == '__main__':
    main()