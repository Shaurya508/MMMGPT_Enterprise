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
from langchain.retrievers.multi_query import MultiQueryRetriever


with open('config.json', 'r') as f:
    config = json.load(f)

# Set environment variables from config file
for key, value in config["ENV_VARS"].items():
    os.environ[key] = value

# Use config for embedding model and chat model
EMBEDDING_MODEL = config["EMBEDDING_MODEL"]
CHAT_MODEL = config["CHAT_MODEL"]

CHUNK_SIZE = config["CHUNK_SIZE"]
CHUNK_OVERLAP = config["CHUNK_OVERLAP"]

# Suppress warnings
warnings.filterwarnings("ignore")

def initialize_models():
    """Initialize and return the embedding and chat models."""
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    chat_model = ChatOllama(model=CHAT_MODEL, temperature=0)
    return embeddings, chat_model

def run_preprocess(ppt_path, ppt_dir, main_img_dir, progress_bar):
    """Runs the preprocess script on the uploaded PPT file with the required directories."""
    command = f"python preprocess.py {ppt_path} {ppt_dir} {main_img_dir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Read output line by line
    for line in iter(process.stdout.readline, ''):
        if line.startswith("PROGRESS:"):
            # Extract the progress value
            progress_value = int(line.strip().split(":")[1])
            # Update the progress bar
            progress_bar.progress(progress_value)
        else:
            # You can handle other output here if needed
            pass

    # Wait for the process to complete
    process.stdout.close()
    process.wait()
    if process.returncode != 0:
        # Handle errors
        error_message = process.stderr.read()
        st.error(f"An error occurred during processing: {error_message}")


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



def user_input_ppt(user_question, retriever, chat_model, EXTRACTED_IMG_DIR, EXTRACTED_CHART_DIR):
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
        return permanent_cache_cleaned[user_question_cleaned], image_address

    prompt_template = """
    Answer the question using the context.If the context and question are not related, reply with "The retrieved context is not relevant to the question."\n
    Context:\n{context}\n
    Question:\n{question}.\n
    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)
    
    docs = retriever.invoke(input=user_question.lower())
    # retriever_from_llm = MultiQueryRetriever.from_llm(
    # retriever=retriever, llm=chat_model
# )
    # docs = retriever_from_llm.invoke(user_question.lower())
    print(docs)
    response = chain({
        "input_documents": docs,
        "question": user_question
    }, return_only_outputs=True)

    # Process images from document metadata
    image_address = []
    if "saturation" in user_question.lower():
        image_address.append('Images\Extracted Images\saturation_curve_ABC.png')
    else:    
        for doc in docs:
            if 'image_paths' in doc.metadata and doc.metadata['image_paths']:
                for img_path in doc.metadata['image_paths']:
                    if 'image' in img_path.lower():
                        image_full_path = os.path.join(EXTRACTED_IMG_DIR, img_path)
                    elif 'chart' in img_path.lower():
                        image_full_path = os.path.join(EXTRACTED_CHART_DIR, img_path)

                    if os.path.exists(image_full_path):
                        image_address.append(image_full_path)
    save_permanent_answer(user_question, response ,image_address)
    return response, image_address if image_address else None

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

def setup_image_directories(PPT_DIR):
    """Set up image directories for the given PPT directory."""
    MAIN_IMG_DIR = os.path.join(PPT_DIR, "Images")
    FULL_SLIDE_IMG_DIR = os.path.join(MAIN_IMG_DIR, "Full Slide Images")
    EXTRACTED_IMG_DIR = os.path.join(MAIN_IMG_DIR, "Extracted Images")
    EXTRACTED_CHART_DIR = os.path.join(MAIN_IMG_DIR, "Extracted Charts")

    # Create the image directories
    os.makedirs(MAIN_IMG_DIR, exist_ok=True)
    os.makedirs(FULL_SLIDE_IMG_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_IMG_DIR, exist_ok=True)
    os.makedirs(EXTRACTED_CHART_DIR, exist_ok=True)

    return MAIN_IMG_DIR, FULL_SLIDE_IMG_DIR, EXTRACTED_IMG_DIR, EXTRACTED_CHART_DIR


# Define the maximum number of free queries
QUERY_LIMIT = 100

# Initialize session state for tracking the number of queries, conversation history, suggested questions, and authentication
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'suggested_question' not in st.session_state:
    st.session_state.suggested_question = ""

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'generate_response' not in st.session_state:
    st.session_state.generate_response = False

if 'chat' not in st.session_state:
    st.session_state.chat = ""

if 'ppt_processed' not in st.session_state:
    st.session_state.ppt_processed = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = ""

if 'current_ppt_dir' not in st.session_state:
    st.session_state.current_ppt_dir = ""

if 'db' not in st.session_state:
    st.session_state.db = None

if 'retriever' not in st.session_state:
    st.session_state.retriever = None

if 'conversation_history_ppt' not in st.session_state:
    st.session_state.conversation_history_ppt = []

if 'generate_response_ppt' not in st.session_state:
    st.session_state.generate_response_ppt = False



def create_ui():
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
                    st.session_state.current_user = email.split("@")[0].lower()
                    st.rerun()
                else:
                    st.error("Invalid email or password. Please try again.")
        return
    
    USER_DATA_PPT_DIR = "User_Data_PPT"
    USER_DIR = os.path.join(USER_DATA_PPT_DIR, st.session_state.current_user)
    os.makedirs(USER_DIR, exist_ok=True)

    # Initialize models
    embeddings, chat_model = initialize_models()


    # Create two tabs
    tab1, tab2 = st.tabs(["Chat with MMM PPT", "Chat with Domain"])

    
    with tab1:

        chat_container1 = st.container()

        if not st.session_state.ppt_processed:
            # Present the two options
            #option = st.radio("Select an option:", ("Chat with a new MMM PPT", "Chat with your Existing MMM PPTs"))
            # Use columns to align buttons horizontally
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Chat with a new MMM PPT"):
                    st.session_state.option = "Chat with a new MMM PPT"
            with col2:
                if st.button("📁 Chat with your Existing MMM PPTs"):
                    st.session_state.option = "Chat with your Existing MMM PPTs"

            # Retrieve the selected option
            option = st.session_state.get('option', None)

            if option == "Chat with a new MMM PPT":
                # Initial PPT upload and processing stage
                uploaded_ppt = st.file_uploader(
                    "Upload your PPT",
                    type="pptx",
                    label_visibility="collapsed",
                    help="Drag and drop your PPT here or browse files."
                )

                if uploaded_ppt is not None:

                    ppt_name = uploaded_ppt.name.replace(" ", "_").split(".")[0]
                    PPT_DIR = os.path.join(USER_DIR, ppt_name)
                    # Create necessary directories for the PPT and its images
                    os.makedirs(PPT_DIR, exist_ok=True)

                    MAIN_IMG_DIR, FULL_SLIDE_IMG_DIR, EXTRACTED_IMG_DIR, EXTRACTED_CHART_DIR = setup_image_directories(PPT_DIR)

                    ppt_save_path = os.path.join(PPT_DIR, uploaded_ppt.name.replace(" ", "_"))

                    DB_NAME = "PPT_FAISS_INDEX"
                    db_save_path = os.path.join(PPT_DIR, DB_NAME)

                    # Save uploaded file
                    with open(ppt_save_path, "wb") as f:
                        f.write(uploaded_ppt.getbuffer())

                    # Create a progress bar
                    progress_bar = st.progress(0)

                    # Process the PPT and update the progress bar
                    run_preprocess(ppt_save_path, PPT_DIR, MAIN_IMG_DIR, progress_bar)
                    
                    # Load database and create retriever
                    st.session_state.db = FAISS.load_local(db_save_path, embeddings, allow_dangerous_deserialization=True)
                    st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={'k': 5})
                    st.session_state.current_ppt_dir = PPT_DIR
                    # Reset conversation history
                    st.session_state.conversation_history_ppt = []
                    st.session_state.chat = ""
                    st.success("PPT has been processed!")
                    st.session_state.ppt_processed = True
                    st.rerun()

            elif option == "Chat with your Existing MMM PPTs":
                # List all PPT folders inside USER_DIR
                ppt_folders = [name for name in os.listdir(USER_DIR) if os.path.isdir(os.path.join(USER_DIR, name))]

                if ppt_folders:
                    selected_ppt = st.selectbox("Select a PPT:", ppt_folders)
                    if st.button("Load Selected PPT"):
                        # Set PPT_DIR to the selected folder
                        PPT_DIR = os.path.join(USER_DIR, selected_ppt)
                        # Update Image directories to be inside the PPT_DIR
                        MAIN_IMG_DIR, FULL_SLIDE_IMG_DIR, EXTRACTED_IMG_DIR, EXTRACTED_CHART_DIR = setup_image_directories(PPT_DIR)
                        # Set the database path
                        DB_NAME = "PPT_FAISS_INDEX"
                        db_save_path = os.path.join(PPT_DIR, DB_NAME)

                        # Load the database and create retriever
                        st.session_state.db = FAISS.load_local(db_save_path, embeddings, allow_dangerous_deserialization=True)
                        st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={'k': 5})
                        st.session_state.current_ppt_dir = PPT_DIR
                        # Reset conversation history
                        st.session_state.conversation_history_ppt = []
                        st.session_state.chat = ""
                        st.success(f"PPT '{selected_ppt}' has been loaded!")
                        st.session_state.ppt_processed = True
                        st.rerun()
                else:
                    st.write("No existing PPTs found. Please upload a new PPT.")

        if st.session_state.ppt_processed:
            PPT_DIR = st.session_state.current_ppt_dir
            # Update Image directories to be inside the PPT_NAME_DIR
            MAIN_IMG_DIR, FULL_SLIDE_IMG_DIR, EXTRACTED_IMG_DIR, EXTRACTED_CHART_DIR = setup_image_directories(PPT_DIR)



            with chat_container1:
                if st.session_state.conversation_history_ppt == []:
                    col1, col2 = st.columns([1, 8])
                    with col1:
                        st.image('download.png', width=30)
                    with col2:
                        st.write("Hello, I am MMMGPT Enterprise from Aryma Labs. How can I help you?")
            
            for idx, (q, r, displayed_image) in enumerate(st.session_state.conversation_history_ppt):
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
                    st.session_state.generate_response_ppt = True

            if st.session_state.generate_response_ppt and question:
                if st.session_state.query_count >= QUERY_LIMIT:
                    st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
                else:
                    with st.spinner("Generating response..."):
                        response, image_address = user_input_ppt(
                                user_question=question,
                                retriever=st.session_state.retriever,
                                chat_model=chat_model,
                                EXTRACTED_IMG_DIR=EXTRACTED_IMG_DIR,
                                EXTRACTED_CHART_DIR=EXTRACTED_CHART_DIR                            
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
                        st.session_state.conversation_history_ppt.append((question, output_text ,displayed_image))
                        st.session_state.generate_response_ppt = False
                        st.session_state.query_count += 1
                        st.rerun()
        st.markdown("---")
        if st.button("🚪 Exit", key='exit_button_tab1'):
            # Reset session states related to Tab 1
            st.session_state.ppt_processed = False
            st.session_state.current_ppt_dir = ""
            st.session_state.db = None
            st.session_state.retriever = None
            st.session_state.conversation_history_ppt = []
            st.session_state.generate_response_ppt = False
            st.session_state.chat = ""
            st.experimental_rerun()  # Refresh the app to reflect changes





    with tab2:
        chat_container = st.container()
        with chat_container:
            if st.session_state.conversation_history == []:
                col1, col2 = st.columns([1, 8])
                with col1:
                    st.image('download.png', width=30)
                with col2:
                    st.write("Hello, I am MMMGPT Enterprise from Aryma Labs. How can I help you?")
    
        for idx, (q, r, suggested_questions, language) in enumerate(st.session_state.conversation_history):
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
                    st.write(r)
    
                target_language = st.selectbox(
                    'Select target language', options=list(LANGUAGES.keys()), key=f'target_language_{idx}'
                )
    
                if st.button('Translate', key=f'translate_button_{idx}'):
                    if target_language:
                        if target_language == "English":
                            st.write(r1)
                        else:
                            if language != LANGUAGES[target_language]:
                                translated_text = translate(clean_text(r1), from_lang="en", to_lang=LANGUAGES[target_language])
                                added_text = translate("For more details, please visit", from_lang="en", to_lang=LANGUAGES[target_language])
                            else:
                                translated_text = clean_text(r)
                                if language != "en":
                                    added_text = translate("For more details, please visit", from_lang="en", to_lang=language)
                                else:
                                    added_text = "For more details, please visit"
                            st.write(translated_text + "\n\n" + added_text + ": " + post_link)
    
                if urls:
                    post_link = urls[0]
                    url = get_image_link(post_link)
                    if url is not None:
                        try:
                            response = requests.get(url)
                            img = Image.open(BytesIO(response.content))
                            st.image(img, use_column_width=True)
                        except UnidentifiedImageError:
                            pass
                        
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
    
            # Create columns for better checkbox layout and use session state for persistence
            col1, col2, col3 = st.columns(3)
    
            if 'cpg_fmcg' not in st.session_state:
                st.session_state.cpg_fmcg = False
            if 'retail' not in st.session_state:
                st.session_state.retail = False
            if 'ecommerce' not in st.session_state:
                st.session_state.ecommerce = False
            if 'energy' not in st.session_state:
                st.session_state.energy = False
            if 'sports' not in st.session_state:
                st.session_state.sports = False
            if 'qsr' not in st.session_state:
                st.session_state.qsr = False
            if 'banking_finance' not in st.session_state:
                st.session_state.banking_finance = False
    
            with col1:
                st.session_state.cpg_fmcg = st.checkbox('CPG/FMCG', value=st.session_state.cpg_fmcg)
                st.session_state.retail = st.checkbox('Retail', value=st.session_state.retail)
                st.session_state.ecommerce = st.checkbox('Ecommerce', value=st.session_state.ecommerce)
    
            with col2:
                st.session_state.energy = st.checkbox('Energy', value=st.session_state.energy)
                st.session_state.sports = st.checkbox('Sports', value=st.session_state.sports)
    
            with col3:
                st.session_state.qsr = st.checkbox('QSR', value=st.session_state.qsr)
                st.session_state.banking_finance = st.checkbox('Banking/Finance', value=st.session_state.banking_finance)
    
            if submit_button and question:
                st.session_state.generate_response = True
    
        if st.session_state.generate_response and question:
            if st.session_state.query_count >= QUERY_LIMIT:
                st.warning("You have reached the limit of free queries. Please consider our pricing options for further use.")
            else:
                with st.spinner("Generating response..."):
                    # Call the appropriate user_input function based on the selected checkbox
                    if st.session_state.cpg_fmcg:
                        response, docs, suggested_questions, language = user_input1(question)
                    elif st.session_state.energy:
                        response, docs, suggested_questions, language = user_input2(question)
                    elif st.session_state.qsr:
                        response, docs, suggested_questions, language = user_input3(question)
                    elif st.session_state.retail:
                        response, docs, suggested_questions, language = user_input4(question)
                    elif st.session_state.sports:
                        response, docs, suggested_questions, language = user_input5(question)
                    elif st.session_state.banking_finance:
                        response, docs, suggested_questions, language = user_input6(question)
                    elif st.session_state.ecommerce:
                        response, docs, suggested_questions, language = user_input7(question)
    
                    output_text = response.get('output_text', 'No response')
                    st.session_state.chat += str(output_text)
                    st.session_state.conversation_history.append((question, output_text, suggested_questions, language))
                    st.session_state.suggested_question = ""
                    st.session_state.query_count += 1
                    st.session_state.generate_response = False
                    st.rerun()




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
