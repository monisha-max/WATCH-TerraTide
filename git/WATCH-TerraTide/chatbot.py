

# #READING AND SAVING IT IN A FILE, IN REALTIME WE USE ACTAUL VECTOR DATABASE LIKE PINECONE 

# import os
# import streamlit as st
# from langchain_openai import OpenAI as OpenAILLM
# from langchain_openai import OpenAIEmbeddings as LLMEmbeddings # Updated import
# from langchain.chains import RetrievalQAWithSourcesChain as QAChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
# from langchain_community.document_loaders import UnstructuredURLLoader as URLDataLoader
# from langchain_community.vectorstores import FAISS as FaissIndex
# from dotenv import load_dotenv
# import time

# def load_api_key(auth_file):
#     with open(auth_file, 'r') as file:
#         for line in file:
#             if line.startswith('OPENAI_API_KEY'):
#                 return line.strip().split('=')[1]
#     return None

# # Load OpenAI API key
# openai_api_key = load_api_key('.auth')

# # Check if the API key is loaded
# if openai_api_key is None:
#     raise ValueError("OpenAI API key not found in .auth file")


# st.set_page_config(page_title="Intellitax Research Tool", layout="wide")

# def set_background_color():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-color: #76ABAE;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
# set_background_color()

# # Load environment variables
# load_dotenv()

# st.title("🔍 Intellitax Research Tool")

# st.sidebar.header("🔗 Enter URLs Here")
# input_urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
# start_processing = st.sidebar.button("Submit")

# #Placeholder for real-time status updates
# status_container = st.container()

# faiss_directory = "faiss_store"

# # Initialize language model with set parameters
# language_model = OpenAILLM(api_key=openai_api_key, temperature=0.9, max_tokens=500)
# llm_embeddings = LLMEmbeddings(api_key=openai_api_key)

# expected_minimum_chunks = 10

# def split_text(loaded_data):
#     # Primary delimiters
#     primary_delimiters = ['\n\n', '\n', '.', ',']
#     # Fallback delimiters
#     fallback_delimiters = [';', ' ', '|']

#     doc_splitter = TextSplitter(separators=primary_delimiters, chunk_size=1000)
#     split_documents = doc_splitter.split_documents(loaded_data)

#     # If splitting fails or returns too few segments, try fallback delimiters
#     if not split_documents or len(split_documents) < expected_minimum_chunks:
#         doc_splitter = TextSplitter(separators=fallback_delimiters, chunk_size=1000)
#         split_documents = doc_splitter.split_documents(loaded_data)

#     return split_documents

# if start_processing:
#     status_container.info("Processing URLs...")
#     # Load and process data from provided URLs
#     try:
#         # Load and process data from provided URLs
#         url_loader = URLDataLoader(urls=input_urls)
#         loaded_data = url_loader.load()

#         # Modify Text Splitter as per your data structure
#         # doc_splitter = TextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
#         split_documents = split_text(loaded_data)

#         # Creating embeddings and FAISS index
        
#         vectorindex_openai = FaissIndex.from_documents(split_documents, llm_embeddings)

#         # Save FAISS index
#         vectorindex_openai.save_local("faiss_store")
#         status_container.success("URLs processed successfully!")
#     except Exception as e:
#         status_container.error(f"Error processing URLs: {e}")

# user_query = st.text_input("🔍 Type your query here")

# if user_query:
#     # Check if the FAISS index exists and load it
#     if os.path.isdir(faiss_directory):
#         loaded_faiss_index = FaissIndex.load_local(faiss_directory, llm_embeddings, allow_dangerous_deserialization=True)

#         # Setting up the QA Chain with the loaded index
#         qa_chain = QAChain.from_llm(llm=language_model, retriever=loaded_faiss_index.as_retriever())

#         # Retrieving the answer to the user's query
#         query_result = qa_chain({"question": user_query}, return_only_outputs=True)

#         # Displaying answer and sources
#         st.subheader("Answer")
#         st.write(query_result["answer"])

#         # Extracting and displaying sources, if available
#         result_sources = query_result.get("sources", "")
#         if result_sources:
#             st.subheader("Sources")
#             st.write(result_sources.replace("\n", ", "))
#     else:
#         status_container.error("FAISS index not found. Please process the URLs first.")

# #to start practicing on google doc
# #how to appraoch 

# import os
# import requests
# import streamlit as st
# from langchain_openai import OpenAI as OpenAILLM
# from langchain_openai import OpenAIEmbeddings as LLMEmbeddings  # Updated import
# from langchain.chains import RetrievalQAWithSourcesChain as QAChain
# from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
# from langchain_community.document_loaders import UnstructuredURLLoader as URLDataLoader
# from langchain_community.vectorstores import FAISS as FaissIndex
# from dotenv import load_dotenv
# from urllib.parse import urlparse
# from langchain.llms import OpenAI

# # Set Streamlit page configuration
# st.set_page_config(page_title="ClimaBot Tool", layout="wide")

# # Load environment variables
# load_dotenv()

# # Load NewsAPI key from environment variables
# news_api_key = os.getenv("NEWS_API_KEY")

# # URL validation function to avoid processing invalid URLs
# def validate_url(url):
#     try:
#         result = urlparse(url)
#         return all([result.scheme, result.netloc])
#     except ValueError:
#         return False

# # Preprocess the user query (lowercasing and removing punctuation)
# def preprocess_query(query):
#     return query.lower().strip().replace('?', '')

# # Expanded keywords related to climate and weather
# weather_keywords = [
#     "weather", "climate", "temperature", "rainfall", "humidity",
#     "storm", "air quality", "precipitation", "greenhouse gases", 
#     "flood", "drought", "tornado", "hurricane", "wildfire", "snowfall", 
#     "global warming", "heat wave", "ozone layer", "carbon footprint", 
#     "sea level rise", "pollution", "renewable energy", "sustainability", 
#     "wind patterns", "monsoon", "environment", "arctic ice", "extreme weather"
# ]

# # Fetch news articles from NewsAPI based on weather and climate-related keywords
# def fetch_weather_climate_news():
#     query = " OR ".join(weather_keywords)  # Join keywords with OR for broader coverage
#     url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={news_api_key}"
#     response = requests.get(url)
#     return response.json()



# # Split text into smaller chunks with semantic boundaries
# def split_text(loaded_data):
#     primary_delimiters = ['\n\n', '\n', '.', '?', '!']
#     fallback_delimiters = [';', ' ', '|']
#     doc_splitter = TextSplitter(separators=primary_delimiters, chunk_size=600)
#     split_documents = doc_splitter.split_documents(loaded_data)

#     return split_documents

# # Load OpenAI API key
# openai_api_key = os.getenv("OPENAI_API_KEY")
# if openai_api_key is None:
#     raise ValueError("OpenAI API key not found")

# # Set background color
# def set_background_color():
#     st.markdown(
#         """
#         <style>
#         .stApp {
#             background-color: #76ABAE;
#         }
#         </style>
#         """,
#         unsafe_allow_html=True
#     )

# set_background_color()

# st.title("ClimaBot Tool")

# # Sidebar for URL input
# st.sidebar.header("🔗 Enter URLs Here")
# input_urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
# start_processing = st.sidebar.button("Submit")

# # Placeholder for real-time status updates
# status_container = st.container()

# faiss_directory = "faiss_store"

# # Initialize language model and embeddings with set parameters
# language_model = OpenAI(api_key=openai_api_key, temperature=0.9, max_tokens=500)
# llm_embeddings = LLMEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")

# expected_minimum_chunks = 10

# # Process the URLs when the 'Submit' button is clicked
# if start_processing:
#     valid_urls = [url for url in input_urls if validate_url(url)]
#     if not valid_urls:
#         status_container.error("Please enter valid URLs.")
#     else:
#         status_container.info("Processing URLs...")

#         try:
#             # Load and process data from valid URLs
#             url_loader = URLDataLoader(urls=valid_urls)
#             loaded_data = url_loader.load()

#             # Split loaded data into chunks
#             split_documents = split_text(loaded_data)

#             # Create embeddings and FAISS index
#             vectorindex_openai = FaissIndex.from_documents(split_documents, llm_embeddings)
#             vectorindex_openai.save_local(faiss_directory)

#             status_container.success("URLs processed successfully!")
#         except Exception as e:
#             status_container.error(f"Error processing URLs: {e}")

# # Accept user query input
# user_query = st.text_input("🔍 Type your query here")

# # Process the query when provided
# if user_query:
#     # Preprocess the user query for better matching
#     user_query = preprocess_query(user_query)

#     # Check if the FAISS index exists and load it
#     if os.path.isdir(faiss_directory):
#         try:
#             loaded_faiss_index = FaissIndex.load_local(faiss_directory, llm_embeddings, allow_dangerous_deserialization=True)

#             # Setting up the QA Chain with the loaded FAISS index
#             retriever = loaded_faiss_index.as_retriever(search_k=10)
#             qa_chain = QAChain.from_llm(llm=language_model, retriever=retriever)

#             # Retrieve the answer to the user's query
#             query_result = qa_chain({"question": user_query}, return_only_outputs=True)

#             # Display the answer
#             st.subheader("Answer")
#             st.write(query_result["answer"])

#             # Display the source of the answer
#             sources = query_result.get("sources", "")
#             if sources:
#                 st.subheader("Sources")
#                 st.write(sources.replace("\n", ", "))

#             # Fetch related news articles for weather and climate
#             news_response = fetch_weather_climate_news()

#             # Display related news articles
#             st.subheader("Related News Articles")
#             if news_response.get("articles"):
#                 for i, article in enumerate(news_response["articles"][:10]): 
#                     st.write(f"**{article['title']}** - {article['source']['name']}")
#                     st.write(f"[Read More]({article['url']})")
#             else:
#                 st.write("No related news articles found.")
#         except Exception as e:
#             status_container.error(f"Error loading FAISS index or processing query: {e}")
#     else:
#         status_container.error("FAISS index not found. Please process the URLs first.")

import os
import requests
import streamlit as st
from langchain.llms import OpenAI
# from langchain.embeddings import OpenAIEmbeddings  # Correct import
from langchain.embeddings.openai import OpenAIEmbeddings  # Possible correct import

from langchain.chains import RetrievalQAWithSourcesChain as QAChain
from langchain.text_splitter import RecursiveCharacterTextSplitter as TextSplitter
from langchain.document_loaders import UnstructuredURLLoader  # Correct loader
from langchain.vectorstores import FAISS as FaissIndex
from dotenv import load_dotenv
from urllib.parse import urlparse

# Set Streamlit page configuration
st.set_page_config(page_title="ClimaBot Tool", layout="wide")

# Load environment variables
load_dotenv()

# Load NewsAPI key from environment variables
news_api_key = os.getenv("NEWS_API_KEY")

# URL validation function to avoid processing invalid URLs
def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# Preprocess the user query (lowercasing and removing punctuation)
def preprocess_query(query):
    return query.lower().strip().replace('?', '')

# Expanded keywords related to climate and weather
weather_keywords = [
    "weather", "climate", "temperature", "rainfall", "humidity",
    "storm", "air quality", "precipitation", "greenhouse gases", 
    "flood", "drought", "tornado", "hurricane", "wildfire", "snowfall", 
    "global warming", "heat wave", "ozone layer", "carbon footprint", 
    "sea level rise", "pollution", "renewable energy", "sustainability", 
    "wind patterns", "monsoon", "environment", "arctic ice", "extreme weather"
]

# Fetch news articles from NewsAPI based on weather and climate-related keywords
def fetch_weather_climate_news():
    if news_api_key is None:
        st.error("NewsAPI key is not found. Please check your environment variables.")
        return {}
    query = " OR ".join(weather_keywords)
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={news_api_key}"
    response = requests.get(url)
    return response.json()

# Split text into smaller chunks with semantic boundaries
def split_text(loaded_data):
    primary_delimiters = ['\n\n', '\n', '.', '?', '!']
    fallback_delimiters = [';', ' ', '|']
    doc_splitter = TextSplitter(separators=primary_delimiters, chunk_size=600)
    split_documents = doc_splitter.split_documents(loaded_data)
    return split_documents

# Load OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OpenAI API key not found")

# Set background color
def set_background_color():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #76ABAE;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background_color()

st.title("ClimaBot Tool")

# Sidebar for URL input
st.sidebar.header("🔗 Enter URLs Here")
input_urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
start_processing = st.sidebar.button("Submit")

# Placeholder for real-time status updates
status_container = st.container()

faiss_directory = "faiss_store"
if not os.path.exists(faiss_directory):
    os.makedirs(faiss_directory)

# Initialize language model and embeddings with set parameters
language_model = OpenAI(api_key=openai_api_key, temperature=0.9, max_tokens=500)
llm_embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-ada-002")

# Process the URLs when the 'Submit' button is clicked
if start_processing:
    valid_urls = [url for url in input_urls if validate_url(url)]
    if not valid_urls:
        status_container.error("Please enter valid URLs.")
    else:
        status_container.info("Processing URLs...")
        try:
            # Load and process data from valid URLs
            url_loader = UnstructuredURLLoader(urls=valid_urls)
            loaded_data = url_loader.load()

            # Split loaded data into chunks
            split_documents = split_text(loaded_data)

            # Create embeddings and FAISS index
            vectorindex_openai = FaissIndex.from_documents(split_documents, llm_embeddings)
            vectorindex_openai.save_local(faiss_directory)

            status_container.success("URLs processed successfully!")
        except Exception as e:
            status_container.error(f"Error processing URLs: {e}")

# Accept user query input
user_query = st.text_input("🔍 Type your query here")

# Process the query when provided
if user_query:
    user_query = preprocess_query(user_query)
    if os.path.isdir(faiss_directory):
        try:
            loaded_faiss_index = FaissIndex.load_local(faiss_directory, llm_embeddings, allow_dangerous_deserialization=True)
            retriever = loaded_faiss_index.as_retriever(search_k=10)
            qa_chain = QAChain.from_llm(llm=language_model, retriever=retriever)
            query_result = qa_chain({"question": user_query}, return_only_outputs=True)

            st.subheader("Answer")
            st.write(query_result["answer"])

            sources = query_result.get("sources", "")
            if sources:
                st.subheader("Sources")
                st.write(sources.replace("\n", ", "))

            news_response = fetch_weather_climate_news()
            st.subheader("Related News Articles")
            if news_response.get("articles"):
                for i, article in enumerate(news_response["articles"][:10]):
                    st.write(f"**{article['title']}** - {article['source']['name']}")
                    st.write(f"[Read More]({article['url']})")
            else:
                st.write("No related news articles found.")
        except Exception as e:
            status_container.error(f"Error loading FAISS index or processing query: {e}")
    else:
        status_container.error("FAISS index not found. Please process the URLs first.")
