#OPENAI_API_KEY=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
#Nk50opwMr51Ig1Ra
#mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
from flask import Flask, render_template, request, redirect
import requests
from pymongo import MongoClient
import openai
#from sentence_transformers import SentenceTransformer
#import faiss
import requests
#from bs4 import BeautifulSoup

# def scrape_health_advice(url):
#     page = requests.get(url)
#     soup = BeautifulSoup(page.content, 'html.parser')
    
#     advice_paragraphs = soup.find_all('p')
#     advice_text = [p.get_text() for p in advice_paragraphs]
#     return advice_text

# model = SentenceTransformer('all-MiniLM-L6-v2')  # BERT-based model to create embeddings
# scraped_data = ['Some text about heatwave safety...', 'Another text on hydration...']

# #Convert documents to vectors
# document_embeddings = model.encode(scraped_data)

# # Build the FAISS index
# index = faiss.IndexFlatL2(384)  # 384 is the dimensionality of the model output
# index.add(document_embeddings)
# def retrieve_relevant_info(query):
#     query_embedding = model.encode([query])
    
#     # Search in the FAISS index for the nearest document
#     D, I = index.search(query_embedding, k=5)  # k is the number of documents to retrieve
#     relevant_docs = [scraped_data[i] for i in I[0]]
#     return relevant_docs



app = Flask(__name__)

# MongoDB Atlas connection
client = MongoClient('mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['heat_tolerance_db']
users_collection = db['users']

# Weather API Key
API_KEY = 'bf1a8422d92b4eedb2d143836242408'

# OpenAI API Key
openai.api_key = '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'

# Function to fetch weather data
def fetch_weather_data(city):
    response = requests.get(f'http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}&aqi=no')
    return response.json()

# Function to generate OpenAI response for personalized advice
def generate_openai_advice(age, health_conditions, activity_level, temp, humidity, rainfall):
    prompt = f"""Based on the details below, generate a fun and interactive heatwave safety guide:

- Age: {age}
- Health conditions: {health_conditions}
- Activity level: {activity_level}
- Current temperature: {temp}°C
- Humidity: {humidity}%
- Rainfall: {rainfall} mm

Give your suggestions in a fun and engaging way, as if you're talking directly to the user. Include:
1. **Quick Hydration Tips**: Offer exciting hydration advice tailored to their age and health conditions (e.g., “Sip your way through the heat!”).
2. **Clothing Style Suggestions**: Keep it stylish but comfortable—suggest breathable and sun-friendly clothing based on the weather and activity level (e.g., “Time for those cool shades!”).
3. **Outdoor Activity Modifications**: Suggest ways to safely enjoy outdoor activities or what to avoid (e.g., “Maybe dodge that midday sun and opt for an evening stroll instead!”).

Your response should be concise and interactive, making the advice exciting and fun for the user. Each tip should be practical and easy to follow. Prioritize hydration, clothing, and activity suggestions. Limit your response to just 5 tips, and make sure they feel friendly and engaging!

Make it exciting, concise, and friendly!
give more emojis pleaseeeeeeeeee"""

    
    # OpenAI ChatCompletion API using GPT-3.5-turbo or GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # You can also use "gpt-4" if available
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing personalized health safety advice."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        n=1,
        stop=None,
        temperature=0.7
    )

    return response['choices'][0]['message']['content']

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST':
#         age = request.form['age']
#         health_conditions = request.form.getlist('health_conditions[]')  # Handle multiple selections
#         other_health_condition = request.form.get('other_health_condition_input', '')

#         # If "Other" is selected, append the custom health condition to the list
#         if 'Other' in health_conditions and other_health_condition:
#             health_conditions.append(other_health_condition)
        
#         activity_level = request.form['activity_level']
#         city = request.form['city']
#         email = request.form['email']

#         # Fetch weather data
#         weather_data = fetch_weather_data(city)
#         temp = weather_data['current']['temp_c']
#         humidity = weather_data['current']['humidity']
#         rainfall = weather_data['current']['precip_mm']

#         # Join health conditions into a single string
#         health_conditions_str = ', '.join(health_conditions)

#         # Generate personalized advice using OpenAI
#         personalized_advice = generate_openai_advice(age, health_conditions_str, activity_level, temp, humidity, rainfall)

#         # Save user data to MongoDB Atlas
#         user_data = {
#             'age': age,
#             'health_conditions': health_conditions,
#             'activity_level': activity_level,
#             'city': city,
#             'email': email
#         }
#         users_collection.insert_one(user_data)

#         # Pass the weather data and advice to the success page
#         return render_template('success.html', advice=personalized_advice, temp=temp, humidity=humidity, rainfall=rainfall, city=city)

#     return render_template('home.html')
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = request.form['age']
        health_conditions = request.form.getlist('health_conditions[]')
        other_health_condition = request.form.get('other_health_condition_input', '')

        if 'Other' in health_conditions and other_health_condition:
            health_conditions.append(other_health_condition)
        
        activity_level = request.form['activity_level']
        city = request.form['city']
        email = request.form['email']

        # Fetch weather data
        weather_data = fetch_weather_data(city)
        temp = weather_data['current']['temp_c']
        humidity = weather_data['current']['humidity']
        rainfall = weather_data['current']['precip_mm']

        # Join health conditions into a single string
        health_conditions_str = ', '.join(health_conditions)

        # Generate personalized advice using OpenAI
        personalized_advice = generate_openai_advice(age, health_conditions_str, activity_level, temp, humidity, rainfall)

        # Save user data to MongoDB Atlas
        user_data = {
            'age': age,
            'health_conditions': health_conditions,
            'activity_level': activity_level,
            'city': city,
            'email': email
        }
        users_collection.insert_one(user_data)

        # Pass weather data and advice to the success page
        return render_template('success.html', advice=personalized_advice, temp=temp, humidity=humidity, rainfall=rainfall, city=city)

    return render_template('home.html')




@app.route('/success')
def success():
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
