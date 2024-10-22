from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import json
import os
# from PyPDF2 import PdfReader #used it before now using tesseract
# import requests
# from bs4 import BeautifulSoup
# from dotenv import load_dotenv
# import google.generativeai as genai
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
import pandas as pd
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
# from langchain.chains import ConversationChain
# from pdf2image import convert_from_path
# from PIL import Image
import streamlit as st
# import pytesseract
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
import time
import re
# from trial import translate
# import langid
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
# Load the Google API key from the .env file
# load_dotenv()
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
# Load the Google API key from the .env file
# load_dotenv()
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# sec_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"]=sec_key
# Function to log in to LinkedIn
def linkedin_login(email, password , driver):
    driver.get("https://www.linkedin.com/login")
    
    # Find the username/email field and send the email
    email_field = driver.find_element(By.ID, "username")
    email_field.send_keys(email)
    
    # Find the password field and send the password
    password_field = driver.find_element(By.ID, "password")
    password_field.send_keys(password)
    
    # Submit the form
    password_field.send_keys(Keys.RETURN)
    
    # Wait for a bit to allow login to complete
    time.sleep(5)

def clean_text(text):
    return re.sub(r'[?.]', '', text).strip().lower()

# Function to scrape LinkedIn post content
def scrape_linkedin_post(url , driver):
    # Open the LinkedIn post URL
    driver.get(url)
    
    # Wait for the content to load
    time.sleep(5)
    
    # Find the main content of the post
    # Note: The actual class names and structure may vary, so you might need to inspect the LinkedIn post's HTML to get the accurate class names or ids
    post_content = driver.find_element(By.CLASS_NAME, 'feed-shared-update-v2__description')
    
    # Extract and return the text content
    if post_content:
        return post_content.text.encode('ascii', 'ignore').decode('ascii')
    else:
        return "Could not find the main content of the post."



def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    article_content = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in article_content])
    return text.encode('ascii', 'ignore').decode('ascii')

def extract_text_from_pdf(pdf_path):
    # Convert PDF to images
    pages = convert_from_path(pdf_path, 300)

    # Iterate through all the pages and extract text
    extracted_text = ''
    for page_number, page_data in enumerate(pages):
        # Perform OCR on the image
        text = pytesseract.image_to_string(page_data)
        extracted_text += f"Page {page_number + 1}:\n{text}\n"
    return extracted_text
#     with open(pdf_path, "rb") as pdfFile:
#         pdfReader = PdfReader(pdfFile)
#         numPages = len(pdfReader.pages)
#         all_text = ""
#         for page_num in range(numPages):
#             page = pdfReader.pages[page_num]
#             text = page.extract_text()
#             if text:
#                 all_text += text.encode('ascii', 'ignore').decode('ascii') + "\n"
#     return all_text

def extract_code_from_github(raw_url):
    # text = extract_text_from_url('https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R')
    # print(text)
    # URL of the raw content of the R script on GitHub
    # raw_url = "https://raw.githubusercontent.com/facebookexperimental/Robyn/main/demo/demo.R"

    # Send a GET request to the raw content URL
    response = requests.get(raw_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the content of the response
        code = response.text

        # Print the scraped code
        return code
    # else:
    #     print(f"Failed to retrieve the URL. Status code: {response.status_code}")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks, batch_size=100):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_embeddings = []
    
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        batch_embeddings = embeddings.embed_documents(batch)
        text_embeddings.extend(zip(batch, batch_embeddings))
    
    vector_store = FAISS.from_embeddings(text_embeddings, embedding=embeddings)
    new_db = FAISS.load_local("Faiss_Index_MMMGPT", embeddings, allow_dangerous_deserialization=True)
    new_db.merge_from(vector_store)
    new_db.save_local("Faiss_Index_MMMGPT")
    return vector_store

# Define the path for the permanent cache file


# Initialize an empty list to store conversation history
conversation_history = []

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    questions_db = FAISS.load_local("faiss_index_questions", embeddings, allow_dangerous_deserialization=True)
    def clean_text(text):
        return re.sub(r'[?.]', '', text).strip().lower()
    language, confidence = langid.classify(user_question)
    print(language)
    if(language != "en"):
        user_question = translate(user_question , language , "en")
    # New code
    print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        # time.sleep(5)
        suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, suggested_questions ,language
    
    prompt_template = """
    you are MMMGPT devloped by Aryma labs to help users on Market Mix Modelling(MMM) , Now you have to chat with the user.Be friendly and Give good responses.
    If the question is general type like "Hi" or "Hello" just don't focus on the context and links. 
    the context given is from more important to less important from top to bottom.
    Try to understand the context and then give detailed answers as much as possible. Don't answer if answer is not from the context but answer to general question like "Hi" and "Hello" etc .
    provide every answer in detailed explanation and easy words to make easy for the User. Don't you ever give wrong links !
    Also, provide one link given in the context(of substack or LinkedIn) only in the following way in the end of the Answer or Provide Pdf Link when question is asked as Summary of Research paper. 
    "For more details visit" : link given in the context only \n
    Context:\n{context}?\n
    Question:\n{question} + Explain in detail.\n
    Answer:
    """
    # prompt_template = """ Summarize the context given as the research paper given in the question and after that 
    # Write like this -> "For more details visit" : (PDF link) given in the context only \n
    # Context:\n{context}?\n
    # Question:\n{question} + Explain in detail.\n
    # Answer :
    # """
    

    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7)
    model = ChatOllama(model="llama3.2:1b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    new_db1 = FAISS.load_local("Faiss_Index_MMMGPT", embeddings, allow_dangerous_deserialization=True)
    # new_db2 = FAISS.load_local("faiss_index2", embeddings, allow_dangerous_deserialization=True)
    
    new_db3 = FAISS.load_local("faiss_index_added", embeddings, allow_dangerous_deserialization=True)
    new_db.merge_from(new_db3)
    # new_db2.merge_from(new_db1)

    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # mq_retriever1 = MultiQueryRetriever.from_llm(retriever = new_db1.as_retriever(search_kwargs={'k': 5}), llm = model)
    # mq_retriever2 = MultiQueryRetriever.from_llm(retriever = new_db2.as_retriever(search_kwargs={'k': 4}), llm = model)


    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    # docs2 = mq_retriever1.get_relevant_documents(query=user_question)
    # docs2 = mq_retriever2.get_relevant_documents(query=user_question)

    # Fast technique
    docs1 = new_db1.similarity_search(query=user_question, k = 5)
    docs2 = new_db.similarity_search(query=user_question, k=5)
    suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
   
    # docs = new_db.similarity_search(query=user_question, k=3)  # Get similar text from the database with the query
    response = chain({"input_documents": docs1+docs2, "question": user_question}, return_only_outputs=True)
    # save_permanent_answer(user_question, response)
    return response , docs1 , suggested_questions , language

    # response  = conversation_chain({"question": user_question})
    # return(response.get("answer"))

def user_input1(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_CPG_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None

def user_input2(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_ENERGY_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None

def user_input3(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_QSR_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None


# Define the path for the permanent cache file
PERMANENT_CACHE_FILE = "Cache_MMMGPT_mini.json"

# Load the permanent cache from the file if it exists
def load_permanent_cache():
    if os.path.exists(PERMANENT_CACHE_FILE):
        with open(PERMANENT_CACHE_FILE, "r") as file:
            return json.load(file)
    return {}

# Save the permanent cache to the file
def save_permanent_cache(cache):
    with open(PERMANENT_CACHE_FILE, "w") as file:
        json.dump(cache, file)

# Initialize the permanent cache
permanent_cache = load_permanent_cache()

def save_permanent_answer(question, answer):
    permanent_cache[question] = answer
    save_permanent_cache(permanent_cache)

def delete_permanent_cache_item(question):
    if question in permanent_cache:
        del permanent_cache[question]
        save_permanent_cache(permanent_cache)
        print(f"Deleted question '{question}' from the permanent cache.")
    else:
        print(f"Question '{question}' not found in the permanent cache.")


def user_input4(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_RETAIL_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None

def user_input5(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_SPORTS_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None

def user_input6(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_FINANCE_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None

def user_input7(user_question):
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = OllamaEmbeddings(model="zxf945/nomic-embed-text:latest")
    # questions_db = FAISS.load_local("faiss_index_questions", embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
    # def clean_text(text):
        # return re.sub(r'[?.]', '', text).strip().lower()
    # language, confidence = langid.classify(user_question)
    # print(language)
    # if(language != "en"):
        # user_question = translate(user_question , language , "en")
    # New code
    # print(user_question)
    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    if user_question_cleaned in permanent_cache_cleaned:
        time.sleep(5)
        # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
        return permanent_cache_cleaned[user_question_cleaned], None, None ,None
    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    prompt_template = """
    Answer the question from the context given below and Don't aswer if you don't know the answer from context.
    Context:\n{context}?\n
    Question:\n{question} \n
    Answer:
    """
    # model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.5)
    model = ChatOllama(model="llama3.2:3b", temperature=0.7)

    # repo_id="mistralai/Mistral-7B-Instruct-v0.2"
    # model=HuggingFaceEndpoint(repo_id=repo_id,max_length=128,temperature=0.7,token=sec_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
      # Model for creating vector embeddings
    new_db = FAISS.load_local("FAISS_INDEX_ECOMMERCE_DOMAIN", embeddings, allow_dangerous_deserialization=True)  # Load the previously saved vector db
    # mq_retriever = MultiQueryRetriever.from_llm(retriever = new_db.as_retriever(search_kwargs={'k': 5}), llm = model )
    # docs1 = mq_retriever.get_relevant_documents(query=user_question)
    docs1 = new_db.similarity_search(user_question , k = 10)
    # questions_db = FAISS.load_local('FAISS_INDEX_Questions' , embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") , allow_dangerous_deserialization=True)
    # suggested_questions = questions_db.similarity_search(query=user_question, k = 5)
    response = chain({"input_documents": docs1, "question": user_question}, return_only_outputs=True)
    save_permanent_answer(user_question , response)
    return response , docs1, None , None




def load_in_db():
    file_path = 'Article_links_substack.xlsx'
    df = pd.read_excel(file_path, header=None)
    url_text_chunks = []

    # for url in df[0]:
    #     article_text = extract_text_from_url(url)
    #     text_chunks = get_text_chunks(article_text)
    #     for chunk in text_chunks:
    #         url_text_chunks.append(f"Substack URL: {url}\n{chunk}")


    # pdf_folder = 'PDFs'

    # # Get a list of all PDF files in the folder
    # pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # # Process each PDF file
    # for pdf in pdf_files:
    #     pdf_path = os.path.join(pdf_folder, pdf)
        # article_text = extract_text_from_pdf(pdf_path)
    #     text_chunks = get_text_chunks(article_text)
    #     for chunk in text_chunks:
    #         url_text_chunks.append(f"Pdf Link : {pdf}\n{chunk}")
    # # Github stuff
    # Github_links =  ['https://github.com/facebookexperimental/Robyn/blob/main/demo/demo.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/allocator.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/auxiliary.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/calibration.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/checks.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/clusters.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/convergence.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/data.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/exports.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/imports.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/inputs.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/json.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/model.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/model.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/outputs.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/pareto.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/plots.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/refresh.R','https://github.com/facebookexperimental/Robyn/blob/main/R/R/response.R' , 'https://github.com/facebookexperimental/Robyn/blob/main/R/R/transformation.R' ,'https://github.com/facebookexperimental/Robyn/blob/main/R/R/zzz.R']
    # for url in Github_links :
    #     article_text = extract_code_from_github(url)
    #     text_chunks = get_text_chunks(article_text)
    #     for chunk in text_chunks:
    #         url_text_chunks.append(f"Github Link : {url}\n{chunk}")
       
    # Research papers stuff
    # research_papers_folder = 'MMM research papers'

    # # Get a list of all PDF files in the folder
    # research_files = [f for f in os.listdir(research_papers_folder) if f.endswith('.pdf')]

    # # Process each PDF file
    # for pdf in research_files:
    #     pdf_path = os.path.join(research_papers_folder, pdf)
    #     article_text = extract_text_from_pdf(pdf_path)
    #     text_chunks = get_text_chunks(article_text)
    #     for chunk in text_chunks:
    #         url_text_chunks.append(f"Pdf Link : {pdf}\n{chunk}")
    driver = webdriver.Chrome()

    linkedin_email = "shauryamishra120210@gmail.com"
    linkedin_password = "Mishra@123"
    # Log in to LinkedIn
    linkedin_login(linkedin_email, linkedin_password , driver)
    # file_path = 'MMMGPT_linkedin_blogs.xlsx'
    # df = pd.read_excel(file_path, header=None)
    # links = df.iloc[:, 0].tolist()
    links = ['https://www.linkedin.com/posts/venkat-raman-analytics_attribution-causation-in-the-last-few-activity-7226469002993131522-WgnV?utm_source=share&utm_medium=member_desktop']

    for linkedin_post_url in links:
        post_text = scrape_linkedin_post(linkedin_post_url , driver)
        text_chunks = get_text_chunks(post_text)
        # image_address = extract_next_image_url(linkedin_post_url , driver)
        for chunk in text_chunks:
            url_text_chunks.append(f"Linkedin URL : {linkedin_post_url}\n{chunk}")

    get_vector_store(url_text_chunks)

def main():
    load_in_db()

if __name__ == "__main__":
    main()