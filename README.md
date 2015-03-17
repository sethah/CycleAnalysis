#Ghost Rider
###A novel prediction for cycling performance.

##Project Summary
This app is used to make predictions for cyclist's performances on any course based on past data (taken from [Strava](https://strava.com)). The app uses predictions on past rides to develop a novel performance metric for cyclists. Users can add themselves or other users to *_any ride_* to see how they stack up against their peers. A dynamic, in-ride pacing tool also provides instant feedback on a rider's performance and allows the rider to adjust their pace dynamically, and intelligently.

###Data collection and storage
**model_dev.ipynb, StravaAPI.py, StravaDB.py**
The StravaAPI module can be used to connect to and query Strava's API, as long as the required registration credentials exist. The StravaDB module can be used to create, structure, and store the API data in a MySQL database.

Strava calls a logged workout or ride "activities." Each user has their own activities, which can only be retrieved with an access token for that user. Thus, each user that uses the Cyclitics app must register with the app and provide an access token. This access token is stored in the athletes table in the database. The database is structured as follows:

####Athletes

+------------+-----------+------+-----+---------+-------+
| Field      | Type      | Null | Key | Default | Extra |
+------------+-----------+------+-----+---------+-------+
| id         | int(11)   | NO   | PRI | NULL    |       |
| firstname  | char(20)  | NO   |     | NULL    |       |
| lastname   | char(20)  | NO   |     | NULL    |       |
| sex        | char(1)   | NO   |     | NULL    |       |
| city       | char(30)  | NO   |     | NULL    |       |
| state      | char(10)  | NO   |     | NULL    |       |
| country    | char(40)  | NO   |     | NULL    |       |
| access_key | char(100) | NO   |     | NULL    |       |
+------------+-----------+------+-----+---------+-------+

####Activities

+----------------------+-----------+------+-----+-------------------+-------+
| Field                | Type      | Null | Key | Default           | Extra |
+----------------------+-----------+------+-----+-------------------+-------+
| id                   | int(11)   | NO   | PRI | NULL              |       |
| start_dt             | timestamp | NO   |     | CURRENT_TIMESTAMP |       |
| timezone             | char(40)  | YES  |     | NULL              |       |
| city                 | char(40)  | YES  |     | NULL              |       |
| country              | char(40)  | YES  |     | NULL              |       |
| start_longitude      | double    | NO   |     | NULL              |       |
| start_latitude       | double    | NO   |     | NULL              |       |
| elapsed_time         | int(11)   | YES  |     | NULL              |       |
| distance             | double    | NO   |     | NULL              |       |
| moving_time          | int(11)   | NO   |     | NULL              |       |
| average_speed        | double    | YES  |     | NULL              |       |
| kilojoules           | double    | YES  |     | NULL              |       |
| max_speed            | double    | YES  |     | NULL              |       |
| name                 | char(100) | YES  |     | NULL              |       |
| total_elevation_gain | double    | YES  |     | NULL              |       |
| athlete_id           | int(11)   | NO   | PRI | NULL              |       |
| fitness10            | double    | YES  |     | NULL              |       |
| fitness30            | double    | YES  |     | NULL              |       |
| frequency10          | int(3)    | YES  |     | NULL              |       |
| frequency30          | int(3)    | YES  |     | NULL              |       |
| athlete_count        | int(3)    | NO   |     | NULL              |       |
+----------------------+-----------+------+-----+-------------------+-------+

####Routes

+----------------------+---------------------+------+-----+-------------------+-----------------------------+
| Field                | Type                | Null | Key | Default           | Extra                       |
+----------------------+---------------------+------+-----+-------------------+-----------------------------+
| id                   | bigint(20) unsigned | NO   | PRI | NULL              | auto_increment              |
| start_dt             | timestamp           | NO   |     | CURRENT_TIMESTAMP | on update CURRENT_TIMESTAMP |
| timezone             | char(40)            | YES  |     | NULL              |                             |
| city                 | char(40)            | YES  |     | NULL              |                             |
| country              | char(40)            | YES  |     | NULL              |                             |
| start_longitude      | double              | NO   |     | NULL              |                             |
| start_latitude       | double              | NO   |     | NULL              |                             |
| distance             | double              | NO   |     | NULL              |                             |
| name                 | char(100)           | YES  |     | NULL              |                             |
| total_elevation_gain | double              | YES  |     | NULL              |                             |
| athlete_id           | int(11)             | NO   |     | NULL              |                             |
| fitness10            | double              | YES  |     | NULL              |                             |
| fitness30            | double              | YES  |     | NULL              |                             |
| frequency10          | int(3)              | YES  |     | NULL              |                             |
| frequency30          | int(3)              | YES  |     | NULL              |                             |
| athlete_count        | int(3)              | NO   |     | NULL              |                             |
+----------------------+---------------------+------+-----+-------------------+-----------------------------+

####Streams

+-------------+------------+------+-----+---------+-------+
| Field       | Type       | Null | Key | Default | Extra |
+-------------+------------+------+-----+---------+-------+
| activity_id | int(11)    | NO   | PRI | NULL    |       |
| athlete_id  | int(11)    | NO   | PRI | NULL    |       |
| time        | double     | NO   | PRI | NULL    |       |
| distance    | double     | NO   |     | NULL    |       |
| grade       | double     | NO   |     | NULL    |       |
| altitude    | double     | NO   |     | NULL    |       |
| velocity    | double     | NO   |     | NULL    |       |
| latitude    | double     | NO   |     | NULL    |       |
| longitude   | double     | NO   |     | NULL    |       |
| moving      | tinyint(1) | NO   |     | NULL    |       |
+-------------+------------+------+-----+---------+-------+


###Feature engineering and signal processing
**StravaEffort.py**
It is not a surprise that the most predictive feature of rider velocity is the grade of the road. This feature is provided by Strava, but it is simply the derivative of altitude with respect to distance. A major problem with this is that the altitude observation is innacurate and noisy, which makes the derivative even noisier. Because the grade is noisy and somewhat innacurate, it is more useful to use a filtered version of the grade. Exactly how heavily filtered the grade ought to be is an interesting question, but as the filter is applied, each point contains more and more information about the past and future points - which is feature engineering in and of itself. It is useful to have a feature that is an aggregate of past and future points (a moving average).

Another useful feature is to calculate a moving average that only contains information about the past n miles of the ride. This feature serves as a relative fatigue factor (e.g. has the rider been climbin constantly for the last one mile, or have they been descending?). 

Some seasonal factors are also important. Rider fitness is quantified by a 10 and 30 day fitness, where both the frequency of the activities and the difficulty of the rider's activities are considered.

Features are created in the StravaActivity.make_df() method, which constructs a feature/target dataframe for the current activity.

###Model development
**model_dev.ipynb, StravaEffort.py, StravaUser.py**
Model development involves testing out different regression models and understanding the tradeoffs for each. Models were primarily evaluated by considering the r-squared value for the test sets. Entire activities were withheld in a test set - roughly 20% of a user's activities are withheld. The remaining 80% of activities were combined and then a train test split was performed on each with a 75% train and a 25% test split. Random forest, gradient boosting, support vector, and linear regression models were tested, with random forest and gradient boosting models achieving the best results. While random forest and gradient boosting perform similarly, the gradient boosting models require less memory and can be stored and opened more easily than random forest models. Stored models are opened and used for predictions regularly when the users use the cyclitics app, so having a model that takes up less space is important.

The models are trained when a user visits the site in the fit() page in views.py.


###Visualization and web development
**app/**
The web app provides a user interface for riders to provide their Strava data to the app and then view their predictions on their own dashboard. Each user can go to their dashboard and view all their past rides and compare with their predictions. They can also upload new routes (rides) in the form of gpx files which provide distance, altitude, and lat/lng points. They can then view their predictions on the new route. Any rider can be added to any ride (past or new), and they will see their prediction on that ride (regardless if they have done the ride before).

The ride can be played back, and the riders will move along the google polyline path on the map while their cursors move across the altitude profile plot. The plot displays the time difference between the rider who's dashboard it is and all other riders. If only the rider and the rider's prediction are displayed then the updated prediction is displayed so that the rider can see how their prediction changes as they ride. If more riders are added, then the rider who is in the lead is displayed. 

The views.py controls all flask routes and all html and javascript rendering are kept within the static folder.