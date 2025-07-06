import asyncio
import numpy as np
import faiss
from crawl4ai import AsyncWebCrawler
from quart import Quart, render_template, request, redirect, session
import openai
import requests
from pymongo import MongoClient
import logging
import json
import os

# --------------------------
# Logging Configuration
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --------------------------
# Quart App Setup
# --------------------------
app = Quart(__name__)
app.secret_key = "monishaterratide"  # Needed for session management

# --------------------------
# MongoDB Setup
# --------------------------
client = MongoClient(
    'mongodb+srv://terratide662:Nk50opwMr51Ig1Ra@cluster0.exm1e.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
)
db = client['heat_tolerance_db']
users_collection = db['users']
analytics_collection = db['analytics']
conversation_collection = db['conversations']

# --------------------------
# API Keys
# --------------------------
WEATHER_API_KEY = '9790d903731d40fdae150122243112'
openai.api_key = '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'

# --------------------------
# FAISS Index Persistence
# --------------------------
FAISS_INDEX_FILE = "faiss_index.index"
FAISS_METADATA_FILE = "faiss_metadata.json"

def create_faiss_index():
    dim = 1536  # Dimension used by OpenAI's text-embedding-ada-002
    index = faiss.IndexFlatL2(dim)
    return index

index = create_faiss_index()
metadata_dict = {}

def save_faiss_index():
    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(FAISS_METADATA_FILE, "w") as f:
        json.dump(metadata_dict, f)
    logging.info("FAISS index and metadata saved to disk.")

def load_faiss_index():
    global index, metadata_dict
    if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(FAISS_METADATA_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(FAISS_METADATA_FILE, "r") as f:
            metadata_dict = json.load(f)
        logging.info("FAISS index and metadata loaded from disk.")
    else:
        logging.info("No existing FAISS index found, starting fresh.")

load_faiss_index()

# --------------------------
# Weather Data Agents
# --------------------------
def fetch_weather_data(city):
    try:
        response = requests.get(
            f'http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}&aqi=no'
        )
        response.raise_for_status()
        data = response.json()
        return data if 'current' in data else None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather data: {e}")
        return None

def fetch_weather_forecast(city, days=3):
    """
    Extended Weather Forecast Agent:
    Fetch a multi-day forecast to add context to the advice.
    """
    try:
        response = requests.get(
            f'http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days={days}&aqi=no&alerts=no'
        )
        response.raise_for_status()
        data = response.json()
        return data.get('forecast', {}) if 'forecast' in data else {}
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching weather forecast: {e}")
        return {}

# --------------------------
# Data Enrichment Agent
# --------------------------
def data_enrichment_agent():
    """
    Fetch additional context from external sources.
    For demonstration purposes, we return a static enrichment message.
    """
    enrichment_text = (
        "Additional Health Advisory: Stay informed with local government alerts and "
        "recent research on heatwave impacts. Always check official sources for accurate updates. "
        "Remember: prevention is better than cure! 🚑🌡️"
    )
    return enrichment_text

# --------------------------
# Helper: Get Embeddings
# --------------------------
def get_embeddings(text):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    embedding = response['data'][0]['embedding']
    return np.array([embedding], dtype=np.float32)

# --------------------------
# Helper: Chunk Text
# --------------------------
def chunk_text(text, max_tokens=500):
    words = text.split()
    chunks = []
    current_chunk = []
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# --------------------------
# Store in FAISS
# --------------------------
def store_in_faiss(text, metadata):
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = get_embeddings(chunk)
        faiss.normalize_L2(embedding)
        index.add(embedding)
        current_ntotal = index.ntotal
        meta_copy = metadata.copy()
        meta_copy["text"] = chunk
        # Use string keys for JSON serialization
        metadata_dict[str(current_ntotal - 1)] = meta_copy
    save_faiss_index()

# --------------------------
# Async Scraping Agent
# --------------------------
async def scrape_website(url, work_type="general", work_role=""):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        markdown_text = result.markdown if result else ""
    metadata = {
        "source_url": url,
        "work_type": work_type,
        "work_role": work_role
    }
    store_in_faiss(markdown_text, metadata)
    logging.info(f"Scraped and stored data from {url}")

# --------------------------
# FAISS Search Agent
# --------------------------
def search_faiss(query, desired_work_type="", desired_work_role=""):
    query_embedding = get_embeddings(query)
    faiss.normalize_L2(query_embedding)
    D, I = index.search(query_embedding, k=3)
    logging.info(f"FAISS distances: {D[0]}, indices: {I[0]}")
    relevant_chunks = []
    for idx in I[0]:
        if idx == -1:
            continue
        meta = metadata_dict.get(str(idx), {})
        if desired_work_type and meta.get("work_type") != desired_work_type:
            continue
        if desired_work_role and meta.get("work_role") != desired_work_role:
            continue
        chunk_text_val = meta.get("text", "")
        relevant_chunks.append(chunk_text_val)
    if relevant_chunks:
        return "\n\n".join(relevant_chunks)
    return ""

# --------------------------
# Query Parsing/Intent Recognition Agent
# --------------------------
def parse_query_intent(query):
    """
    Use GPT to parse and extract the main intent and keywords from the user's query.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that extracts the main intent and keywords from a query."
        },
        {
            "role": "user",
            "content": f"Extract the main intent and keywords from the following query: {query}"
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=100,
        temperature=0.5
    )
    return response['choices'][0]['message']['content'].strip()

# --------------------------
# OpenAI Advice Generation Agent
# --------------------------
import time
import openai

def generate_openai_advice(age, health_conditions, activity_level,
                           temp, humidity, rainfall, city,
                           work_type, work_role, faiss_context="",
                           forecast_context=""):
    enrichment = data_enrichment_agent()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a health and safety expert providing ultra-personalized heatwave safety advice. "
                "Below is context from the user's knowledge base (scraped data), extended weather forecast, and additional enrichment data. "
                "Please incorporate relevant points in your final answer.\n\n"
                f"Scraped CONTEXT:\n{faiss_context}\n\n"
                f"Forecast CONTEXT:\n{forecast_context}\n\n"
                f"Enrichment:\n{enrichment}"
            )
        },
        {
            "role": "user",
            "content": f"""
            Age: {age}
            Health Conditions: {health_conditions}
            Activity Level: {activity_level}
            Work Type: {work_type}
            Role: {work_role}
            Current Weather in {city}: {temp}°C, {humidity}% humidity, {rainfall} mm rainfall.

            Based on the above and the provided context, please provide a 5-point heatwave safety guide that includes:
              1. Hydration tips (with a detox drink suggestion tailored to the health conditions)
              2. Clothing & PPE recommendations
              3. Activity adjustments
              4. Special care for the specified health conditions
              5. Location-specific and creative ideas

            Use emojis liberally.
            """
        }
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=messages,
                max_tokens=500,
                temperature=0.85
            )
            return response['choices'][0]['message']['content'].strip()
        except openai.error.APIError as e:
            # Log error and retry after a delay
            logging.error(f"OpenAI APIError on attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)
    raise Exception("Failed to generate advice after several attempts.")


# --------------------------
# Follow-up Response Agent
# --------------------------
def generate_followup_response(followup_question, original_advice):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a health and safety expert. The user received an initial heatwave safety guide and now asks for further clarification or additional tips. "
                "Below is the initial advice provided:\n\n" + original_advice
            )
        },
        {
            "role": "user",
            "content": f"Follow-up Question: {followup_question}\n\nPlease provide additional guidance."
        }
    ]
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        max_tokens=300,
        temperature=0.85
    )
    return response['choices'][0]['message']['content'].strip()

# --------------------------
# User Analytics and Profiling Agent
# --------------------------
def track_user_interaction(email, event, details={}):
    interaction = {
        "email": email,
        "event": event,
        "details": details
    }
    analytics_collection.insert_one(interaction)
    logging.info(f"Tracked interaction for {email}: {event}")

def store_conversation_history(email, conversation_entry):
    entry = {
        "email": email,
        "conversation": conversation_entry
    }
    conversation_collection.insert_one(entry)
    logging.info(f"Stored conversation history for {email}")

# --------------------------
# Routes
# --------------------------
@app.route('/', methods=['GET', 'POST'])
async def home():
    if request.method == 'POST':
        form = await request.form
        name = form.get('name', "")
        age = form.get('age', "")
        email = form.get('email', "")
        city = form.get('city', "")
        work = form.get('work', "")  # "indoor" or "outdoor"
        # Defaults
        work_type = ""
        work_role = ""
        activity_level = ""
        health_conditions_list = []
        if work == "indoor":
            health_conditions_list = form.getlist('health_conditions[]')
            other_health_condition_input = form.get('other_health_condition_input', "").strip()
            if "Other" in health_conditions_list and other_health_condition_input:
                health_conditions_list.remove("Other")
                health_conditions_list.append(other_health_condition_input)
            activity_level = form.get('activity_level', "")
        elif work == "outdoor":
            health_cond_outdoor = form.get('health_conditions_outdoor', "")
            if health_cond_outdoor == "yes":
                health_conditions_list = ["Yes-outdoor"]
            else:
                health_conditions_list = ["No-outdoor"]
            work_type = form.get('work_type', "")
            work_role = form.get('work_role', "")
            activity_level = ""
        # Fetch current weather data
        weather_data = fetch_weather_data(city)
        if not weather_data:
            return await render_template('home.html', error="Could not fetch weather data.")
        temp = weather_data['current']['temp_c']
        humidity = weather_data['current']['humidity']
        rainfall = weather_data['current']['precip_mm']
        # Fetch extended forecast data
        forecast_data = fetch_weather_forecast(city)
        forecast_context = ""
        if forecast_data:
            forecast_days = forecast_data.get('forecastday', [])
            for day in forecast_days:
                date = day.get('date', '')
                day_info = day.get('day', {})
                condition = day_info.get('condition', {}).get('text', '')
                max_temp = day_info.get('maxtemp_c', '')
                forecast_context += f"Date: {date}, Condition: {condition}, Max Temp: {max_temp}°C\n"
        # Construct query for FAISS search
        query = (
            f"City: {city}; Age: {age}; Work: {work}; "
            f"Work Type: {work_type}; Role: {work_role}; "
            f"Health conditions: {', '.join(health_conditions_list)}; "
            f"Activity Level: {activity_level}"
        )
        faiss_context = search_faiss(query, work_type, work_role)
        # Generate personalized advice
        personalized_advice = generate_openai_advice(
            age=age,
            health_conditions=', '.join(health_conditions_list),
            activity_level=activity_level,
            temp=temp,
            humidity=humidity,
            rainfall=rainfall,
            city=city,
            work_type=work_type,
            work_role=work_role,
            faiss_context=faiss_context,
            forecast_context=forecast_context
        )
        # Save user data to MongoDB
        users_collection.insert_one({
            'name': name,
            'age': age,
            'email': email,
            'city': city,
            'work': work,
            'work_type': work_type,
            'work_role': work_role,
            'health_conditions': health_conditions_list,
            'activity_level': activity_level
        })
        # Store initial advice in session and log interaction
        session['original_advice'] = personalized_advice
        track_user_interaction(email, "initial_advice_generated", {"city": city, "work": work})
        return await render_template(
            'success.html',
            advice=personalized_advice,
            temp=temp,
            humidity=humidity,
            rainfall=rainfall,
            city=city
        )
    return await render_template('home.html')

@app.route('/followup', methods=['GET', 'POST'])
async def followup():
    """
    Endpoint for users to ask follow-up questions after receiving initial advice.
    """
    if request.method == 'POST':
        form = await request.form
        followup_question = form.get('followup_question', "")
        original_advice = session.get('original_advice', "")
        email = form.get('email', "")
        if not followup_question or not original_advice:
            return await render_template('followup.html', error="Missing follow-up question or session data.")
        parsed_intent = parse_query_intent(followup_question)
        logging.info(f"Parsed intent: {parsed_intent}")
        followup_response = generate_followup_response(followup_question, original_advice)
        conversation_entry = {
            "followup_question": followup_question,
            "parsed_intent": parsed_intent,
            "followup_response": followup_response
        }
        store_conversation_history(email, conversation_entry)
        track_user_interaction(email, "followup_generated", {"question": followup_question})
        return await render_template(
            'followup.html', 
            original_advice=original_advice,
            followup_response=followup_response
        )
    return await render_template('followup.html')

@app.route('/scrape', methods=['GET'])
async def scrape_and_store():
    """
    Endpoint to scrape a predefined list of URLs and store the data in FAISS.
    """
    logging.info("Starting scraping process...")
    urls = [
        # FOOD AND DRINKS
        "https://urbanwormcompany.com/wp-content/uploads/2018/09/Water-Amounts-in-Fruits-and-Vegetables-Handout-Week-10.pdf",
        "https://www.health.harvard.edu/staying-healthy/using-food-to-stay-hydrated",
        "https://www.mrmed.in/health-library/health-care/top10-summer-drinks",
        # SKIN PROTECTION
        "https://www.aarp.org/entertainment/style-trends/info-2022/best-sunscreen-dark-skin-tones.html",
        "https://rjtcsonline.com/HTMLPaper.aspx?Journal=Research%20Journal%20of%20Topical%20and%20Cosmetic%20Sciences;PID=2015-6-2-1",
        "https://www.health.harvard.edu/blog/protect-your-skin-during-heat-waves-heres-how-202408143066",
        "https://consumeraffairs.nic.in/sites/default/files/file-uploads/ctocpas/sunscreen-13.pdf",
        # MEDICINES STORAGE
        "https://www.bcm.edu/news/heat-medications-dont-mix",
        # MEDICINES
        "https://www2.gnb.ca/content/gnb/en/departments/ocmoh/healthy_environments/content/heat_related_illnesses/medications_and_theheat.html",
        "https://www.cdc.gov/heat-health/hcp/clinical-guidance/heat-and-medications-guidance-for-clinicians.html",
        "https://www.goodrx.com/health-topic/dermatology/avoid-the-sun-if-you-take-these-drugs",
        "https://timesofindia.indiatimes.com/life-style/health-fitness/health-news/common-medications-that-can-increase-risk-of-dehydration-during-intense-heatwave-heres-what-to-do/articleshow/111056291.cms",
        # CLOTHING
        "https://www.rei.com/learn/expert-advice/how-to-dress-for-humidity.html?",
        "https://globaltextilesource.com/news/avoid-wearing-these-fabrics-in-the-heat-and-what-to-wear-instead?",
        "https://fcdrycleaners.com/blog/the-best-fabrics-for-hot-weather/?",
        # OUTDOOR WORKERS (PPE)
        "https://www.magidglove.com/safety-matters/choosing-and-using-ppe/worker-safety-in-extreme-temps?srsltid=AfmBOopi8NNuEoq9asw1oTNXO6F7_69DIqRZtvZluAYDt_ae-XYENJR3",
        "https://www.osha.gov/emergency-preparedness/guides/heat-stress",
        "http://osha.gov/heat-exposure/controls",
        "https://www.dir.ca.gov/dosh/etools/08-006/EWP_workClothing.htm",
        # GENERAL PRECAUTIONS
        "https://www.vectorsolutions.com/resources/blogs/heat-stress-prevention/",
        "https://www.cdc.gov/heat-health/about/index.html",
        "https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(21)01208-3/fulltext",
        "https://www.wbhealth.gov.in/uploaded_files/notice/heat_illnesses.pdf",
        "https://www.weather.gov/safety/heat-during",
        "https://www.cdc.gov/heat-health/media/pdfs/tips-for-people-with-heart-disease-508.pdf",
        "https://www.cdc.gov/heat-health/media/pdfs/tips-for-pregnant-woman-508.pdf",
        "https://www.cdc.gov/heat-health/risk-factors/heat-and-older-adults-aged-65.html",
        "https://www.betterhealth.vic.gov.au/health/healthyliving/how-to-cope-and-stay-safe-in-extreme-heat",
        "https://www.cdph.ca.gov/Programs/CFH/DMCAH/Pages/Health-Topics/Safe-Pregnancies-in-Extreme-Heat.aspx",
        # IMPACTS
        "https://www.nature.com/articles/s41591-024-03395-8",
        "https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-022-08341-3",
        # EXTENDED IMPACTS DUE TO HIGH HEAT: PREGNANT WOMEN
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC6910775/",
        "https://www.pnas.org/doi/full/10.1073/pnas.1806393116",
        "https://www.sciencedirect.com/science/article/pii/S2666667722000757?via%3Dihub#bib0111",
        "https://www.cdc.gov/niosh/docs/mining/UserFiles/works/pdfs/2017-126.pdf",
        "https://www.cdc.gov/niosh/docs/mining/UserFiles/works/pdfs/2017-124.pdf",
        "https://publications.aap.org/pediatrics/article/128/3/e741/30624/Climatic-Heat-Stress-and-Exercising-Children-and",
        "https://bmchealthservres.biomedcentral.com/articles/10.1186/s12913-022-08341-3",
        "https://www.sciencedirect.com/science/article/pii/S0160412018305683?via%3Dihub",
        "https://publications.aap.org/pediatricsinreview/article-abstract/40/3/97/35260/Heat-Related-Illness-in-Children-in-an-Era-of?redirectedFrom=fulltext?autologincheck=redirected",
        "https://www.researchgate.net/publication/251875813_Clothing_insulation_and_temperature_layer_and_mass_of_clothing_under_comfortable_environmental_conditions",
        "https://www.starhealth.in/blog/summer-detox-drinks/",
        "https://www.clinikally.com/blogs/news/15-hydrating-summer-detox-drinks-revitalize-your-health-beat-the-heat?srsltid=AfmBOorJO9iyqETFAVmPKY9SKMTsS68uej5PccEhGH9KAlIZ_hCgoG0o",
        "https://www.dir.ca.gov/dosh/etools/08-006/EWP_workClothing.htm",
        "https://ncdc.mohfw.gov.in/wp-content/uploads/2024/03/NPCCHH_Public-health-advisory_Extreme-heat_Heatwave_2024.pdf",
        "https://www.cdc.gov/niosh/heat-stress/recommendations/",
        "https://ndma.gov.in/Natural-Hazards/Heat-Wave/Dos-Donts",
        "https://tgpwu.org/wp-content/uploads/2024/08/Impact_of_Extreme_Heat_on_Gig_Workers_A_Survey_Report-1.pdf"
    ]
    for url in urls:
        logging.info(f"Scraping URL: {url}")
        await scrape_website(url, "general", "")
    return "Scraped data has been stored in FAISS."

# --------------------------
# Run the App
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
