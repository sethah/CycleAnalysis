import MySQLdb as sql
import pymongo
import traceback

CLIENT = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")
MongoDB = CLIENT.strava

class StravaDB(object):

    def __init__(self):
        self.db = sql.connect(host="127.0.0.1",
                              user="hendris",
                              passwd="abc123",
                              db="hendris$strava")
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
                UNIQUE(id)
                );
            """
        self.execute(q, fetch=False)

    def create_activities_table(self):
        q = """ CREATE TABLE activities
                (
                id INT PRIMARY KEY        NOT NULL,
                start_dt DATE             NOT NULL,
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

    def move_activities(self):
        activities = MongoDB.activities.find()
        for a in activities:
            d = {'id': a['id'],
                 'start_dt': datetime.strptime(a['start_date_local', '%Y-%m-%dT%H:%M:%SZ')
                 'timezone': a['timezone'],
                 'city': a['location_city'],
                 'country': a['location_country'],
                 'start_longitude': a['start_longitude'],
                 'start_latitude': a['start_latitude'],
                 
            }
            # self.insert_values('athletes', d)
            print d

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
            print d



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