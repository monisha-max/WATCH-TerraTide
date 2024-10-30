from flask import Flask, render_template, request
import requests
import openai

app = Flask(__name__)

# OpenAI API Key Setup
openai.api_key = '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'

# Air Quality API configuration
API_KEY = 'bf1a8422d92b4eedb2d143836242408'
API_URL = 'https://api.weatherapi.com/v1/current.json'

# Function to fetch air quality data based on location
def get_air_quality_data(location):
    response = requests.get(API_URL, params={'key': API_KEY, 'q': location, 'aqi': 'yes'})
    if response.status_code == 200:
        data = response.json()
        air_quality = data['current']['air_quality']
        # Calculate AQI value based on specific pollutants
        aqi_value = calculate_aqi(air_quality)
        return {
            'pm2_5': air_quality['pm2_5'],
            'pm10': air_quality['pm10'],
            'co': air_quality['co'],
            'no2': air_quality['no2'],
            'o3': air_quality['o3'],
            'aqi_value': aqi_value
        }
    else:
        return None

# Function to calculate AQI based on pollutants (simplified version)
def calculate_aqi(air_quality):
    # Assume a simplified calculation where AQI is determined by the worst pollutant
    pm2_5_aqi = air_quality['pm2_5'] / 12.0 * 100
    pm10_aqi = air_quality['pm10'] / 50.0 * 100
    co_aqi = air_quality['co'] / 9.0 * 100
    no2_aqi = air_quality['no2'] / 53.0 * 100
    o3_aqi = air_quality['o3'] / 70.0 * 100
    return max(pm2_5_aqi, pm10_aqi, co_aqi, no2_aqi, o3_aqi)

FEATURE_WEIGHTS = {
    'green_wall': 0.10,
    'urban_agriculture': 0.15,
    'mesh_filters': 0.05,
    'rooftop_garden': 0.20,
    'vertical_garden': 0.10,
    'reflective_materials': 0.05,
    'permeable_pavement': 0.05,
    'bio_swales': 0.15,
    'green_roof': 0.20,
    'none': 0.0
}

def calculate_green_space_percentage(area_sqft, floors, air_quality, building_type, features):
    # Base factors (can vary by city or region)
    base_factor_residential = 0.18  # Residential: 18% green space per 1000 sq.ft
    base_factor_commercial = 0.12   # Commercial: 12% green space per 1000 sq.ft
    base_factor_industrial = 0.10   # Industrial: 10% green space

    # Select base factor based on building type
    base_factor = base_factor_residential if building_type == 'residential' else base_factor_commercial

    # Adjustments based on air quality
    pm2_5_factor = 0.03 if air_quality['pm2_5'] > 50 else 0
    pm10_factor = 0.025 if air_quality['pm10'] > 100 else 0
    co_factor = 0.02 if air_quality['co'] > 5 else 0
    no2_factor = 0.015 if air_quality['no2'] > 40 else 0
    o3_factor = 0.01 if air_quality['o3'] > 50 else 0
    pollution_adjustment = 1 + pm2_5_factor + pm10_factor + co_factor + no2_factor + o3_factor

    # Calculate total building area
    total_building_area = area_sqft * floors

    # Calculate green space percentage with pollution adjustment
    green_space_percentage = base_factor * pollution_adjustment * 100  # Convert to percentage

    # Always provide suggestions
    optimization_suggestions = get_openai_suggestions()

    # Adjust green space needed based on selected features
    if 'none' not in features:
        for feature in features:
            green_space_percentage -= green_space_percentage * FEATURE_WEIGHTS[feature]

    return {'green_space_percentage': green_space_percentage, 'suggestions': optimization_suggestions}

# Function to suggest plants based on air quality (enhanced)
def suggest_plants(air_quality):
    plants = []
    if air_quality['pm2_5'] > 50:
        plants.append('Spider Plant (PM2.5 reducer)')
        plants.append('Peace Lily (PM2.5 reducer)')
    if air_quality['pm10'] > 100:
        plants.append('Areca Palm (PM10 reducer)')
    if air_quality['co'] > 5:
        plants.append('Aloe Vera (CO reducer)')
    if air_quality['no2'] > 40:
        plants.append('Snake Plant (NO2 reducer)')
    if air_quality['o3'] > 50:
        plants.append('English Ivy (O3 reducer)')

    if not plants:
        plants.append('General low-maintenance plants like Ferns, Bamboo Palm, and Golden Pothos')
    
    return plants

def get_openai_suggestions():
    prompt = """
    Suggest alternatives to traditional green space for a building project, such as green walls, rooftop gardens, eco-friendly air-purification systems, reflective materials, water conservation techniques, and other methods to reduce air pollution when green space is decreased, separately line by line.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert in environmental planning and green infrastructure."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    return response['choices'][0]['message']['content']

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/calculate', methods=['POST'])
def calculate():
    location = request.form['location']
    area_sqft = float(request.form['area_sqft'])
    floors = int(request.form['floors'])
    building_type = request.form['building_type']
    selected_features = request.form.getlist('features')  # Get selected features

    # Get air quality data and calculate green space
    air_quality = get_air_quality_data(location)
    if air_quality:
        calculation_result = calculate_green_space_percentage(area_sqft, floors, air_quality, building_type, selected_features)
        green_space_percentage = calculation_result.get('green_space_percentage', 0)
        plant_suggestions = suggest_plants(air_quality)
        openai_suggestions = calculation_result.get('suggestions', get_openai_suggestions())  # Fallback if suggestions are not present

        return render_template('result2.html', 
                               green_space_percentage=green_space_percentage, 
                               air_quality=air_quality, 
                               plants=plant_suggestions, 
                               openai_suggestions=openai_suggestions, 
                               aqi_value=air_quality['aqi_value'])
    else:
        return "Error fetching air quality data. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
