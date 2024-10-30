from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
from geopy.geocoders import Nominatim
import folium

app = Flask(__name__)

# MongoDB connection
client = MongoClient('mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['heatmap_db']
collection = db['heat_data']

# Homepage route - Heat Map Display
@app.route('/home1')
def home1():
    # Fetch data from MongoDB
    heat_data = list(collection.find())

    # Initialize map centered at a default location
    m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    # Add temperature and humidity data to the map
    for data in heat_data:
        lat = data['latitude']
        lon = data['longitude']
        temperature = data.get('temperature')
        humidity = data.get('humidity')
        weather_feeling = data.get('weather_feeling', 'No feedback')  # Default if not provided

        # Set marker color based on temperature range
        if temperature <= 25:
            marker_color = 'blue'  # Cool temperature
        elif 25 < temperature <= 30:
            marker_color = 'green'  # Moderate temperature
        elif 30 < temperature <= 35:
            marker_color = 'orange'  # Warm temperature
        else:
            marker_color = 'red'  # Hot temperature

        popup_info = f"Location: {data['location']}<br>Temperature: {temperature}°C<br>Humidity: {humidity}%<br>Feedback: {weather_feeling}"

        folium.CircleMarker(
            location=[lat, lon],
            radius=7,
            popup=popup_info,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7
        ).add_to(m)

    # Save the map as HTML and pass it to the template
    map_html = m._repr_html_()
    return render_template('home1.html', map_html=map_html)




# Route to submit heat data
@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        location = request.form['location']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        weather_feeling = request.form.get('weather_feeling', '')  # Optional input

        # Geocode the location to get latitude and longitude
        geolocator = Nominatim(user_agent="heatmap")
        location_obj = geolocator.geocode(location)
        lat, lon = location_obj.latitude, location_obj.longitude

        # Insert data into MongoDB
        collection.insert_one({
            'location': location,
            'latitude': lat,
            'longitude': lon,
            'temperature': temperature,
            'humidity': humidity,
            'weather_feeling': weather_feeling
        })

        return redirect(url_for('home1'))

    return render_template('submit.html')



if __name__ == '__main__':
    app.run(debug=True,port=5100)
