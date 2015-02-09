from app import app
from flask import render_template, request, flash
from flask import Flask
import requests


@app.route("/test")
def ride_with_gps():
    pass

@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    url = '''   http://www.ridewithgps.com/find/
                search.json?search%5Bkeywords%5D=&search%5Bstart_location%5D=Portland%2C+OR&search%5Bstart_distance%5D=15&search%5Belevation_max%5D=20000&search%5Belevation_min%5D=0&search%5Blength_max%5D=1200&search%5Blength_min%5D=50&search%5Boffset%5D=0&search%5Blimit%5D=20&search%5Bsort_by%5D=length+asc'''

    routes = ride_with_gps()
    # routes = [6201273, 4591226]
    # flash(routes)
    # return str(routes)
    # route = request.form['route']
    if request.method == 'POST':
        route = int(request.form['routes'])
    else:
        route = 6201273
    # return str(route)
    return render_template('maps.html', display_route=route, routes=routes)

def ride_with_gps():
    key = 'afdcf5b531ac5fe206dc584d4a0646f7'
    url = '''http://www.ridewithgps.com/find/search.json?
    search%5Bkeywords%5D=&search%5Bstart_distance%5D=15&search%5Belevation_max%5D=20000&search%5Belevation_min%5D=0&search%5Blength_max%5D=1200&search%5Blength_min%5D=50&search%5Boffset%5D=0&search%5Blimit%5D=20&search%5Bsort_by%5D=length+asc'''
    url = '''http://www.ridewithgps.com/find/search.json'''
    payload = {'search[start_location]': 'Portland, OR', 
                'search[start_distance]': 15,
                'search[elevation_max]': 2000,
                'search[elevation_min]': 0,
                'search[length_max]': 1200,
                'search[length_min]': 50,
                'search[offset]': 0,
                'search[limit]': 20,
                'search[sort_by]': 'length asc'}
    # url = 'http://www.ridewithgps.com/routes/141014.json'
    response = requests.get(url, params=payload)
    # print response.content
    routes = []
    json = response.json()['results']
    for activity in json:
        if 'trip' in activity:
            routes.append(activity['trip']['route_id'])

    return routes