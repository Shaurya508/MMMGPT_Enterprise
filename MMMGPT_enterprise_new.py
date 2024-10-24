import streamlit as st
import pandas as pd
# from memory import user_input
from ollama123 import user_input,  user_input1 , user_input2 , user_input3, user_input4, user_input5, user_input6 , user_input7
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import requests
import re
from trial import translate
from Levenshtein import distance as levenshtein_distance
import hashlib
import subprocess
import config
import os 
import warnings
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import warnings
from langchain_ollama import ChatOllama, OllamaEmbeddings
import time
import json 

# Set environment variables
for key, value in config.ENV_VARS.items():
    os.environ[key] = value

# Suppress warnings
warnings.filterwarnings("ignore")

# Create necessary directories
os.makedirs(config.PPT_DIRECTORY, exist_ok=True)
os.makedirs(config.MAIN_IMG_DIR, exist_ok=True)
os.makedirs(config.FULL_SLIDE_IMG_DIR, exist_ok=True)
os.makedirs(config.EXTRACTED_IMG_DIR, exist_ok=True)
os.makedirs(config.EXTRACTED_CHART_DIR, exist_ok=True)

def initialize_models():
    """Initialize and return the embedding and chat models."""
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    chat_model = ChatOllama(model=config.CHAT_MODEL, temperature=0)
    return embeddings, chat_model

def run_preprocess(ppt_path):
    """Runs the preprocess script on the uploaded PPT file."""
    command = f"python preprocess.py {ppt_path}"
    subprocess.run(command, shell=True)
    print("pre-prepocessing done !")

def get_image_hash(image_path):
    """Compute the SHA-256 hash of the image file."""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return hashlib.sha256(image_bytes).hexdigest()

# Define the path for the permanent cache file
PERMANENT_CACHE_FILE = "Cache_MMMGPT_Enterprise.json"

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

def save_permanent_answer(question, answer, image_address):
    def clean_text(text):
        return re.sub(r'[?.]', '', text).strip().lower()

    # Clean the question only (not the "Image Address" suffix)
    question_cleaned = clean_text(question)

    permanent_cache[question_cleaned] = answer
    permanent_cache[question_cleaned + "Image Address"] = image_address

    save_permanent_cache(permanent_cache)

def delete_permanent_cache_item(question):
    if question in permanent_cache:
        del permanent_cache[question]
        save_permanent_cache(permanent_cache)
        print(f"Deleted question '{question}' from the permanent cache.")
    else:
        print(f"Question '{question}' not found in the permanent cache.")



def user_input_ppt(user_question, time0, retriever, chat_model):
    """Process user query and return response with relevant documents and images."""
    def clean_text(text):
        return re.sub(r'[?.]', '', text).strip().lower()

    user_question_cleaned = clean_text(user_question)

    # Clean all keys in the permanent_cache, but keep "Image Address" unchanged
    permanent_cache_cleaned = {clean_text(key): value for key, value in permanent_cache.items()}

    # Check if the cleaned question exists in the permanent cache
    if user_question_cleaned in permanent_cache_cleaned:
        # Use cleaned question + "Image Address" to retrieve image addresses
        image_key = user_question_cleaned + "Image Address"
        image_address = permanent_cache.get(image_key, None)  # Use .get() to avoid KeyError
        time.sleep(2)
        return permanent_cache_cleaned[user_question_cleaned], None, None, None, image_address

    prompt_template = """
    You have been given a question and a relevant context. Answer the question using the context. If the context and question are not related, reply with "The retrieved context is not relevant to the question."\n
    Context:\n{context}\n
    Question:\n{question}.\n
    Answer (in detail):
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    
    docs = retriever.invoke(input=user_question.lower())

    time1 = time.time() - time0
    time2 = time.time()
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)
    time3 = time.time() - time2

    # Process images from document metadata
    image_address = []
    for doc in docs:
        if 'image_paths' in doc.metadata and doc.metadata['image_paths']:
            for img_path in doc.metadata['image_paths']:
                if 'image' in img_path.lower():
                    image_full_path = os.path.join(config.EXTRACTED_IMG_DIR, img_path)
                elif 'chart' in img_path.lower():
                    image_full_path = os.path.join(config.EXTRACTED_CHART_DIR, img_path)

                if os.path.exists(image_full_path):
                    image_address.append(image_full_path)
    save_permanent_answer(user_question, response ,image_address)
    return response, None, None, None, image_address if image_address else None


# Define the maximum number of free queries
QUERY_LIMIT = 100

# Initialize session state for tracking the number of queries, conversation history, suggested questions, and authentication
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'conversation_history1' not in st.session_state:
    st.session_state.conversation_history1 = []

if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'generate_response' not in st.session_state:
    st.session_state.generate_response = False

if 'generate_response1' not in st.session_state:
    st.session_state.generate_response1 = False

if 'chat' not in st.session_state:
    st.session_state.chat = ""

if 'ppt_processed' not in st.session_state:
    st.session_state.ppt_processed = False


def clean_text(text):
    # Remove asterisks used for bold formatting
    text = re.sub(r'\*+', '', text)
    # Remove text starting from "For more details"
    text = re.sub(r'For more details.*$', '', text, flags=re.IGNORECASE)
    return text


def get_image_link(article_link, file_path='Article_Links.xlsx'):
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Ensure the columns are named correctly
    df.columns = ['Article Link', 'Image link']

    # Create a dictionary mapping article links to image links
    link_mapping = dict(zip(df['Article Link'], df['Image link']))

    # Find the most similar article link using Levenshtein distance
    most_similar_link = min(df['Article Link'], key=lambda x: levenshtein_distance(x, article_link))
    image_link = link_mapping.get(most_similar_link, "Image link not found")
    
    if image_link == "Image link not found" or image_link == 0:
        return None
    return image_link

def authenticate_user(email):
    # Load the Excel file
    df = pd.read_excel('user.xlsx')
    # Convert the input email to lowercase
    email = email.lower()
    # Convert the emails in the dataframe to lowercase
    df['Email'] = df['Email'].str.lower()
    # Check if the email matches any entry in the file
    user = df[df['Email'] == email]
    if not user.empty:
        return True
    return False

def create_ui():
    # hide_streamlit_style = """
    # <style>
    # #MainMenu {visibility: hidden;}
    # footer {visibility: hidden;}
    # footer:after {content:''; display:block; position:relative; top:2px; color: transparent; background-color: transparent;}
    # .viewerBadge_container__1QSob {display: none !important;}
    # .stActionButton {display: none !important;}
    # ::-webkit-scrollbar {
    #     width: 12px;  /* Keep the width of the scrollbar */
    # }
    # ::-webkit-scrollbar-track {
    #     background: #f1f1f1;
    # }
    # ::-webkit-scrollbar-thumb {
    #     background: #888;
    # }
    # ::-webkit-scrollbar-thumb:hover {
    #     background: #555;
    # }
    # .scroll-icon {
    #     position: fixed;
    #     bottom: 40px;  /* Adjusted the position upwards */
    #     right: 150px;
    #     font-size: 32px;
    #     cursor: pointer;
    #     color: #0adbfc;
    #     z-index: 1000;
    # }
    # </style>
    # <script>
    # function scrollToBottom() {
    #     window.scrollTo(0, 50000);
    # }
    # </script>
    # """

    # st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # st.markdown("<h2 style='text-align: center; color: #0adbfc;'><u>Aryma Labs - MMMGPT Enterprise</u></h2>", unsafe_allow_html=True)
    # st.sidebar.image("Aryma Labs Logo.jpeg")
    # st.sidebar.markdown("<h2 style='color: #08daff;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    # st.sidebar.write("Ask anything about MMM and get accurate answers.")

    # if not st.session_state.authenticated:
    #     st.markdown("<h3 style='color: #4682B4;'>Login</h3>", unsafe_allow_html=True)
    #     with st.form(key='login_form'):
    #         email = st.text_input("Email")
    #         login_button = st.form_submit_button(label='Login')

    #         if login_button:
    #             if authenticate_user(email):
    #                 st.session_state.authenticated = True
    #                 st.experimental_rerun()
    #             else:
    #                 st.error("Invalid email or password. Please try again.")
    #     return

    #     # Initialize models
    # embeddings, chat_model = initialize_models()

    # # Session state for database and retriever
    # if 'db' not in st.session_state:
    #     st.session_state.db = None
    # if 'retriever' not in st.session_state:
    #     st.session_state.retriever = None

    # uploaded_ppt = st.sidebar.file_uploader("Upload your PPT", type="pptx")

    # if uploaded_ppt is not None:
    #     # PPT is uploaded
    #     name = uploaded_ppt.name.replace(" ", "_")
    #     ppt_save_path = os.path.join(config.PPT_DIRECTORY, name)

    #     # Save the uploaded file
    #     with open(ppt_save_path, "wb") as f:
    #         f.write(uploaded_ppt.getbuffer())

    #     st.write("Processing your PPT...")
    #     run_preprocess(ppt_save_path)

    #     # Load the database and create a retriever
    #     st.session_state.db = FAISS.load_local(config.DB_NAME, embeddings, allow_dangerous_deserialization=True)
    #     st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={'k': 3})
    #     st.success("PPT has been processed. You can now ask questions.")

    #     # Once the PPT is processed, show the query interface
    #     if st.session_state.retriever:
    #         col1, col2 = st.columns([4, 1])
    #         with col1:
    #             user_query = st.text_input("Enter your query:", "")
    #         with col2:
    #             get_answer = st.button("Get Answer")

    #         if get_answer and user_query.strip() != "":
    #             with st.spinner("Processing your query..."):
    #                 start_time = time.time()
    #                 try:
    #                     response, docs, time1, time3, image_address = user_input2(
    #                         user_question=user_query,
    #                         time0=start_time,
    #                         retriever=st.session_state.retriever,
    #                         chat_model=chat_model
    #                     )

    #                     # Display results
    #                     st.subheader("Answer:")
    #                     st.write(response.get('output_text', 'No relevant documents found.'))

    #                     if image_address:
    #                         displayed_image_hashes = set()
    #                         for image in image_address:
    #                             image_hash = get_image_hash(image)
    #                             if image_hash not in displayed_image_hashes:
    #                                 st.image(image, caption=f"Retrieved Image: {image}")
    #                                 displayed_image_hashes.add(image_hash)

    #                     if docs:
    #                         st.subheader("Retrieved Documents:")
    #                         for idx, doc in enumerate(docs, start=1):
    #                             with st.expander(f"Document {idx}"):
    #                                 st.write(doc.page_content)

    #                     st.subheader("Timing Information:")
    #                     st.write(f"Retrieving Docs: {time1:.2f} seconds")
    #                     st.write(f"Query Generation: {time3:.2f} seconds")
    #                     st.write(f"Overall Time: {time.time() - start_time:.2f} seconds")

    #                 except Exception as e:
    #                     st.error(f"An error occurred while processing the query: {e}")

    #         elif get_answer:
    #             st.warning("Please enter a query to get an answer.")
    # else:
    #     # Display the conversation history in reverse order to resemble a chat interface
    #     chat_container = st.container()
    #     with chat_container:
    #         if st.session_state.conversation_history == []:
    #             col1, col2 = st.columns([1, 8])
    #             with col1:
    #                 st.image('download.png', width=30)
    #             with col2:
    #                 st.write("Hello, I am MMM GPT from Aryma Labs. How can I help you?")

    #     for idx, (q, r, suggested_questions,language) in enumerate(st.session_state.conversation_history):
    #         st.markdown(f"<p style='text-align: right; color: #484f4f;'><b>{q}</b></p>", unsafe_allow_html=True)
    #         col1, col2 = st.columns([1, 8])
    #         r1 = r
    #         with col1:
    #             st.image('download.png', width=30)
    #         with col2:
    #             LANGUAGES = {
    #                 'Arabic': 'ar', 'Azerbaijani': 'az', 'Catalan': 'ca', 'Chinese': 'zh', 'Czech': 'cs',
    #                 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Finnish': 'fi',
    #                 'French': 'fr', 'German': 'de', 'Greek': 'el', 'Hebrew': 'he', 'Hindi': 'hi',
    #                 'Hungarian': 'hu', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja',
    #                 'Korean': 'ko', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Russian': 'ru',
    #                 'Slovak': 'sk', 'Spanish': 'es', 'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk',
    #                 'Bengali': 'bn'
    #             }
    #             urls = re.findall(r'https?://\S+', r)
    #             if urls:
    #                 post_link = urls[0]
    #             else:
    #                 post_link = ""
    #             # st.write(r)
    #             if(language != "en"):
    #                 r = translate(clean_text(r) , "en" , language)
    #                 st.write(r + "\n")
    #             else:
    #                 st.write(clean_text(r1) + "\n")
    #             if(post_link != ""):
    #                 if(language != "en"):
    #                     st.write(translate("For more details, please visit", from_lang='en', to_lang= language) + ": " + post_link)
    #                 else:
    #                     st.write("For more details, please visit :" + post_link)

    #             # Language selection
    #             target_language = st.selectbox(
    #                 'Select target language', options=list(LANGUAGES.keys()), key=f'target_language_{idx}'
    #             )

    #             if st.button('Translate', key=f'translate_button_{idx}'):
                
    #                 if target_language:
    #                     # Translation
    #                     if(target_language == "English"):
    #                         st.write(r1)
    #                         # st.write("For more details, please visit : " + post_link)
    #                     else:
    #                         if(language != LANGUAGES[target_language]):
    #                             # print(language + "\n" + LANGUAGES[target_language])
    #                             translated_text = translate(clean_text(r1), from_lang= "en" , to_lang=LANGUAGES[target_language])
    #                             added_text = translate("For more details, please visit", from_lang= "en", to_lang=LANGUAGES[target_language])
    #                         else:
    #                             translated_text = clean_text(r)
    #                             if(language != "en"):
    #                                 added_text = translate("For more details, please visit", from_lang= "en", to_lang= language)
    #                             else:
    #                                 added_text = "For more details, please visit"
    #                     # Display the translation
    #                     # st.subheader('Translated Text')
    #                         st.write( translated_text + "\n\n" + added_text + ": " + post_link)

    #             if urls:
    #                 post_link = urls[0]
    #                 url = get_image_link(post_link)  # Define this function based on your need
    #                 if url is not None:
    #                     try:
    #                         response = requests.get(url)
    #                         img = Image.open(BytesIO(response.content))
    #                         st.image(img, use_column_width=True)
    #                     except UnidentifiedImageError:
    #                         pass
    #                 # print("hello")

    #             st.write("Explore Similiar questions :")
    #             for i, suggested_question in enumerate(suggested_questions):
    #                 # Ensure unique keys for buttons
    #                 if(suggested_question.page_content != q):
    #                     if st.button(suggested_question.page_content, key=f"suggested_questions_{idx}_{i}", use_container_width=True):
    #                         st.session_state.suggested_question = suggested_question.page_content
    #                         st.session_state.generate_response = True

    # st.markdown("---")
    # instr = "Ask a question:"
    # with st.form(key='input_form', clear_on_submit=True):
    #     col1, col2 = st.columns([8, 1])
    #     with col1:
    #         if st.session_state.suggested_question:
    #             question = st.text_input(instr, value=st.session_state.suggested_question, key="input_question", label_visibility='collapsed')
    #         else:
    #             question = st.text_input(instr, key="input_question", placeholder=instr, label_visibility='collapsed')
    #     with col2:
    #         submit_button = st.form_submit_button(label='Chat')

    #     # Create columns for better checkbox layout
    #     col1, col2, col3 = st.columns(3)
        
    #     with col1:
    #         cpg_fmcg = st.checkbox('CPG/FMCG')
    #         retail = st.checkbox('Retail')
    #         ecommerce = st.checkbox('Ecommerce')
        
    #     with col2:
    #         energy = st.checkbox('Energy')
    #         sports = st.checkbox('Sports')
        
    #     with col3:
    #         qsr = st.checkbox('QSR')
    #         banking_finance = st.checkbox('Banking/Finance')

    #     if submit_button and question:
    #         st.session_state.generate_response = True
    
    # if st.session_state.generate_response and question:
    #     if st.session_state.query_count >= QUERY_LIMIT:
    #         st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
    #     else:
    #         with st.spinner("Generating response..."):
    #             # response, docs , suggested_questions , language = user_input(question)
    #             # Call the appropriate user_input function based on the selected checkbox
    #             if cpg_fmcg:
    #                 response, docs, suggested_questions, language = user_input1(question)
    #             elif energy:
    #                 response, docs, suggested_questions, language = user_input2(question)
    #             elif qsr:
    #                 response, docs, suggested_questions, language = user_input3(question)
    #             elif retail:
    #                 response, docs, suggested_questions, language = user_input4(question)
    #             elif sports:
    #                 response, docs, suggested_questions, language = user_input5(question)
    #             elif banking_finance:
    #                 response, docs, suggested_questions, language = user_input6(question)
    #             elif ecommerce:
    #                 response, docs, suggested_questions, language = user_input7(question)
    #             print(docs)
    #             output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
    #             st.session_state.chat += str(output_text)
    #             st.session_state.conversation_history.append((question, output_text ,suggested_questions, language))
    #             st.session_state.suggested_question = ""  # Reset the suggested question after submission
    #             st.session_state.query_count += 1  # Increment the query count
    #             st.session_state.generate_response = False
    #             st.experimental_rerun()
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    footer:after {content:''; display:block; position:relative; top:2px; color: transparent; background-color: transparent;}
    .viewerBadge_container__1QSob {display: none !important;}
    .stActionButton {display: none !important;}
    ::-webkit-scrollbar {
        width: 12px;  /* Keep the width of the scrollbar */
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    .scroll-icon {
        position: fixed;
        bottom: 40px;  /* Adjusted the position upwards */
        right: 150px;
        font-size: 32px;
        cursor: pointer;
        color: #0adbfc;
        z-index: 1000;
    }
    </style>
    <script>
    function scrollToBottom() {
        window.scrollTo(0, 50000);
    }
    </script>
    """

    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #0adbfc;'><u>Aryma Labs - MMMGPT Enterprise</u></h2>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg")
    st.sidebar.markdown("<h2 style='color: #08daff;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    st.sidebar.write("Ask anything about MMM and get accurate answers.")

    # User login logic
    if not st.session_state.authenticated:
        st.markdown("<h3 style='color: #4682B4;'>Login</h3>", unsafe_allow_html=True)
        with st.form(key='login_form'):
            email = st.text_input("Email")
            login_button = st.form_submit_button(label='Login')

            if login_button:
                if authenticate_user(email):
                    st.session_state.authenticated = True
                    st.experimental_rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return

    # Initialize models
    embeddings, chat_model = initialize_models()

    # Session state for database and retriever
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None

    # Create two tabs
    tab1, tab2 = st.tabs(["Chat with MMM PPT", "Chat with Domain"])

    # with tab1:
    
    #         # Initialize models
    #     embeddings, chat_model = initialize_models()
        
    #     # Session state for database and retriever
    #     if 'db' not in st.session_state:
    #         st.session_state.db = None
    #     if 'retriever' not in st.session_state:
    #         st.session_state.retriever = None
    #     if 'ppt_processed' not in st.session_state:
    #         st.session_state.ppt_processed = False
    #     if 'queries' not in st.session_state:
    #         st.session_state.queries = []  # To store all queries and responses
    
    #     # Initial PPT upload and processing stage
    #     if not st.session_state.ppt_processed:
    #         uploaded_ppt = st.file_uploader(
    #             "Upload your PPT",
    #             type="pptx",
    #             label_visibility="collapsed",
    #             help="Drag and drop your PPT here or browse files."
    #         )
    
    #         if uploaded_ppt is not None:
    #             name = uploaded_ppt.name.replace(" ", "_")
    #             ppt_save_path = os.path.join(config.PPT_DIRECTORY, name)
                
    #             # Save uploaded file
    #             with open(ppt_save_path, "wb") as f:
    #                 f.write(uploaded_ppt.getbuffer())
                
    #             # st.write("Processing your PPT...")
    #             with st.spinner("Processing your PPT..."):
    #                 # run_preprocess(ppt_save_path)
    #                 time.sleep(1)
                
    #             # Load database and create retriever
    #             st.session_state.db = FAISS.load_local(config.DB_NAME, embeddings, allow_dangerous_deserialization=True)
    #             st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={'k': 3})
    #             st.success("PPT has been processed!")
    #             st.session_state.ppt_processed = True
    #             st.rerun()  # Refresh UI after processing
    
    #     # Query and response stage
    #     if st.session_state.ppt_processed:
    #         # Display previous queries and responses
    #         for query, response in st.session_state.queries:
    #             st.markdown(f"**Q:** {query}")
    #             st.markdown(f"**A:** {response}")
    
    #         # New query input
    #         user_query = st.text_input("Ask your query here", key="user_query")
    #         col1, col2 = st.columns([4, 1])
            
    #         with col1:
    #             generate_response = st.button("Generate Response", key="generate_button")
            
    #         with col2:
    #             exit_ppt = st.button("Exit", key="exit_button")
    
    #         if generate_response and user_query.strip() != "":
    #             with st.spinner("Processing your query..."):
    #                 start_time = time.time()
    #                 try:
    #                     response, docs, time1, time3, image_address = user_input_ppt(
    #                         user_question=user_query,
    #                         time0=start_time,
    #                         retriever=st.session_state.retriever,
    #                         chat_model=chat_model
    #                     )
                        
    #                     # Display the response and keep it in session
    #                     st.markdown(f"**Q:** {user_query}")
    #                     st.markdown(f"**A:** {response.get('output_text', 'No relevant documents found.')}")
                        
    #                     # Store the query and response in session state for persistence
    #                     st.session_state.queries.append((user_query, response.get('output_text', 'No relevant documents found.')))
                        
    #                     if image_address:
    #                         displayed_image_hashes = set()
    #                         displayed_image = []
    #                         for image in image_address:
    #                             image_hash = get_image_hash(image)
    #                             if image_hash not in displayed_image_hashes:
    #                                 st.image(image)
    #                                 displayed_image_hashes.add(image_hash)
    #                                 displayed_image.append(image)
    
    #                 except Exception as e:
    #                     st.error(f"An error occurred while processing the query: {e}")

    #         if exit_ppt:
    #             # Reset all states on exit
    #             st.session_state.ppt_processed = False
    #             st.session_state.db = None
    #             st.session_state.retriever = None
    #             st.session_state.queries = []  # Clear all stored queries
    #             st.rerun()  # Restart the app by rerunning it from the beginning
    with tab1:
        chat_container1 = st.container()
        # Initial PPT upload and processing stage
        if not st.session_state.ppt_processed:
            uploaded_ppt = st.file_uploader(
                "Upload your PPT",
                type="pptx",
                label_visibility="collapsed",
                help="Drag and drop your PPT here or browse files."
            )
    
            if uploaded_ppt is not None:
                name = uploaded_ppt.name.replace(" ", "_")
                ppt_save_path = os.path.join(config.PPT_DIRECTORY, name)
                
                # Save uploaded file
                with open(ppt_save_path, "wb") as f:
                    f.write(uploaded_ppt.getbuffer())
                
                # st.write("Processing your PPT...")
                with st.spinner("Processing your PPT..."):
                    # run_preprocess(ppt_save_path)
                    time.sleep(1)
                
                # Load database and create retriever
                st.session_state.db = FAISS.load_local(config.DB_NAME, embeddings, allow_dangerous_deserialization=True)
                st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={'k': 3})
                st.success("PPT has been processed!")
                st.session_state.ppt_processed = True
                st.rerun()  # Refresh UI after processing

        with chat_container1:
            if st.session_state.conversation_history1 == []:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image('download.png', width=30)
                with col2:
                    st.write("Hello, I am MMMGPT Enterprise from Aryma Labs. How can I help you?")
        
        for idx, (q, r, displayed_image) in enumerate(st.session_state.conversation_history1):
            st.markdown(f"<p style='text-align: right; color: #484f4f;'><b>{q}</b></p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 8])
            with col1:
                st.image('download.png', width=30)
            with col2:
                st.write(r)
                for image_link in displayed_image:
                    st.image(image_link)

        st.markdown("---")
        instr = "Ask a question:"
        with st.form(key='input_form', clear_on_submit=True):
            col1, col2 = st.columns([8, 1])
            with col1:
                if st.session_state.suggested_question:
                    question = st.text_input(instr, value=st.session_state.suggested_question, key="input_question", label_visibility='collapsed')
                else:
                    question = st.text_input(instr, key="input_question", placeholder=instr, label_visibility='collapsed')
            with col2:
                submit_button = st.form_submit_button(label='Chat')
        
        if submit_button and question:
                st.session_state.generate_response1 = True

        if st.session_state.generate_response1 and question:
            if st.session_state.query_count >= QUERY_LIMIT:
                st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
            else:
                with st.spinner("Generating response..."):
                    start_time = time.time()
                    response, docs, time1, time3, image_address = user_input_ppt(
                            user_question=question,
                            time0=start_time,
                            retriever=st.session_state.retriever,
                            chat_model=chat_model
                        )
                    displayed_image = []
                    if image_address:
                            displayed_image_hashes = set()
                            for image in image_address:
                                image_hash = get_image_hash(image)
                                if image_hash not in displayed_image_hashes:
                                    st.image(image)
                                    displayed_image_hashes.add(image_hash)
                                    displayed_image.append(image)
                    output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                    st.session_state.chat += str(output_text)
                    st.session_state.conversation_history1.append((question, output_text ,displayed_image))
                    st.session_state.generate_response1 = False
                    st.experimental_rerun()











    with tab2:
        # st.header("Standard Query with Checkboxes")

        chat_container = st.container()
        with chat_container:
            if st.session_state.conversation_history == []:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image('download.png', width=30)
                with col2:
                    st.write("Hello, I am MMMGPT Enterprise from Aryma Labs. How can I help you?")

        for idx, (q, r, suggested_questions,language) in enumerate(st.session_state.conversation_history):
            st.markdown(f"<p style='text-align: right; color: #484f4f;'><b>{q}</b></p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 8])
            r1 = r
            with col1:
                st.image('download.png', width=30)
            with col2:
                LANGUAGES = {
                    'Arabic': 'ar', 'Azerbaijani': 'az', 'Catalan': 'ca', 'Chinese': 'zh', 'Czech': 'cs',
                    'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Finnish': 'fi',
                    'French': 'fr', 'German': 'de', 'Greek': 'el', 'Hebrew': 'he', 'Hindi': 'hi',
                    'Hungarian': 'hu', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja',
                    'Korean': 'ko', 'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Russian': 'ru',
                    'Slovak': 'sk', 'Spanish': 'es', 'Swedish': 'sv', 'Turkish': 'tr', 'Ukrainian': 'uk',
                    'Bengali': 'bn'
                }
                urls = re.findall(r'https?://\S+', r)
                if urls:
                    post_link = urls[0]
                else:
                    # post_link = ""
                    st.write(r)
                # if(language != "en"):
                #     r = translate(clean_text(r) , "en" , language)
                #     st.write(r + "\n")
                # else:
                #     st.write(clean_text(r1) + "\n")
                # if(post_link != ""):
                #     if(language != "en"):
                #         st.write(translate("For more details, please visit", from_lang='en', to_lang= language) + ": " + post_link)
                #     else:
                #         st.write("For more details, please visit :" + post_link)

                # Language selection
                target_language = st.selectbox(
                    'Select target language', options=list(LANGUAGES.keys()), key=f'target_language_{idx}'
                )

                if st.button('Translate', key=f'translate_button_{idx}'):
                
                    if target_language:
                        # Translation
                        if(target_language == "English"):
                            st.write(r1)
                            # st.write("For more details, please visit : " + post_link)
                        else:
                            if(language != LANGUAGES[target_language]):
                                # print(language + "\n" + LANGUAGES[target_language])
                                translated_text = translate(clean_text(r1), from_lang= "en" , to_lang=LANGUAGES[target_language])
                                added_text = translate("For more details, please visit", from_lang= "en", to_lang=LANGUAGES[target_language])
                            else:
                                translated_text = clean_text(r)
                                if(language != "en"):
                                    added_text = translate("For more details, please visit", from_lang= "en", to_lang= language)
                                else:
                                    added_text = "For more details, please visit"
                        # Display the translation
                        # st.subheader('Translated Text')
                            st.write( translated_text + "\n\n" + added_text + ": " + post_link)

                if urls:
                    post_link = urls[0]
                    url = get_image_link(post_link)  # Define this function based on your need
                    if url is not None:
                        try:
                            response = requests.get(url)
                            img = Image.open(BytesIO(response.content))
                            st.image(img, use_column_width=True)
                        except UnidentifiedImageError:
                            pass
                    # print("hello")

        st.markdown("---")
        instr = "Ask a question:"
        with st.form(key='input_form1', clear_on_submit=True):
            col1, col2 = st.columns([8, 1])
            with col1:
                if st.session_state.suggested_question:
                    question = st.text_input(instr, value=st.session_state.suggested_question, key="input_question1", label_visibility='collapsed')
                else:
                    question = st.text_input(instr, key="input_question1", placeholder=instr, label_visibility='collapsed')
            with col2:
                submit_button = st.form_submit_button(label='Chat')
    
            # Create columns for better checkbox layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cpg_fmcg = st.checkbox('CPG/FMCG')
                retail = st.checkbox('Retail')
                ecommerce = st.checkbox('Ecommerce')
            
            with col2:
                energy = st.checkbox('Energy')
                sports = st.checkbox('Sports')
            
            with col3:
                qsr = st.checkbox('QSR')
                banking_finance = st.checkbox('Banking/Finance')
    
            if submit_button and question:
                st.session_state.generate_response = True
        
        if st.session_state.generate_response and question:
            if st.session_state.query_count >= QUERY_LIMIT:
                st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
            else:
                with st.spinner("Generating response..."):
                    # response, docs , suggested_questions , language = user_input(question)
                    # Call the appropriate user_input function based on the selected checkbox
                    if cpg_fmcg:
                        response, docs, suggested_questions, language = user_input1(question)
                    elif energy:
                        response, docs, suggested_questions, language = user_input2(question)
                    elif qsr:
                        response, docs, suggested_questions, language = user_input3(question)
                    elif retail:
                        response, docs, suggested_questions, language = user_input4(question)
                    elif sports:
                        response, docs, suggested_questions, language = user_input5(question)
                    elif banking_finance:
                        response, docs, suggested_questions, language = user_input6(question)
                    elif ecommerce:
                        response, docs, suggested_questions, language = user_input7(question)
                    print(docs)
                    output_text = response.get('output_text', 'No response')  # Extract the 'output_text' from the response
                    st.session_state.chat += str(output_text)
                    st.session_state.conversation_history.append((question, output_text ,suggested_questions, language))
                    st.session_state.suggested_question = ""  # Reset the suggested question after submission
                    st.session_state.query_count += 1  # Increment the query count
                    st.session_state.generate_response = False
                    st.experimental_rerun()



    # Scroll to bottom icon
    st.markdown("""
        <div class="scroll-icon" onclick="scrollToBottom()">⬇️</div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>Powered by: Aryma Labs</p>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #A9A9A9;'>MMMGPT can occasionally make mistakes. Please refer to the links given at the end of each response for more relevant and accurate results.</p>", unsafe_allow_html=True)
    
# Main function to run the app
def main():
    create_ui()

if __name__ == "__main__":
    main()
