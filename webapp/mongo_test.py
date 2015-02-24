import pymongo

client = pymongo.MongoClient("mongodb://sethah:abc123@ds049161.mongolab.com:49161/strava")

db = client.strava
table = db.activities

localclient = pymongo.MongoClient()
localdb = localclient.mydb
localtable = localdb.activities

# table.remove()
# for item in localtable.find():
#     table.insert(item)

print table.find()[0]