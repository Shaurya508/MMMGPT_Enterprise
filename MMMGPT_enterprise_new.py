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
    st.markdown("<h2 style='text-align: center; color: #0adbfc;'><u>Aryma Labs - MMM GPT</u></h2>", unsafe_allow_html=True)
    st.sidebar.image("Aryma Labs Logo.jpeg")
    st.sidebar.markdown("<h2 style='color: #08daff;'>Welcome to Aryma Labs</h2>", unsafe_allow_html=True)
    st.sidebar.write("Ask anything about MMM and get accurate answers.")

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

    with st.sidebar.expander("Popular Questions", expanded=False):
        suggested_questions = [
            "What is Marketing Mix Modeling?",
            "What are Contribution Charts?",
            "What is the ideal R square range for MMM?",
            "What are the ideal adstock range of media channels?",
            "How MMMs can be calibrated and validated?",
            "Why Frequentist MMM is better than Bayesian MMM?"
        ]

        for i, question in enumerate(suggested_questions):
            if st.button(question, key=f"popular_question_{i}", use_container_width=True):
                st.session_state.suggested_question = question
                st.session_state.generate_response = True

    with st.sidebar.expander("Summarize research papers ", expanded=False):
        research_papers = [
            "Proving Efficacy of Marketing Mix Modeling (MMM) through the Difference in Difference (DID) technique",
            "Investigation Of Marketing Mix Models’ Business Error Using KL Divergence And Chebyshev’s Inequality",
            "Calibrating Marketing Mix Models through Probability Integral Transform (PIT) Residuals",
            "Granger Causality-A possible Feature Selection Method in Marketing Mix Modeling MMM",
            "The Concept of the Marketing Mix - Neil Borden",
            "Understanding Advertising Adstock Transformations",
            "Marketing Mix Modeling (MMM) -Concepts and Model Interpretation",
            "Challenges And Opportunities In Media Mix Modeling",
            "Optimization of Media Strategy via Marketing Mix Modeling in Retailing",
            "Hierarchical Marketing Mix Models with Sign Constraints",
            "Media mix modeling - A Monte Carlo simulation study",
            "Packaging Up Media Mix Modeling: An Introduction to Robyn's Open-Source Approach",
            "Bias Correction For Paid Search In Media Mix Modeling",
            "Marketing Mix Modeling for the Tourism Industry: A Best Practices Approach",
            "Using mixture-amount modeling to optimize the advertising media mix and quantify cross-media synergy for specific target groups",
            "MARKETING MIX THEORETICAL ASPECTS",
            "The Evaluation of Marketing Mix Elements: A Case Study",
            "The effect of marketing mix in attracting customers: Case study of Saderat Bank in Kermanshah Province",
            "MODELING MARKETING MIX",
            "The 4S Web-Marketing Mix model",
            "The Marketing Mix Revisited: Towards the 21st Century Marketing",
            "THE ULTIMATE THEORY OF THE MARKETING MIX: A PROPOSAL FOR MARKETERS AND MANAGERS",
            "Marketing Mix Modeling Using PLS-SEM, Bootstrapping the Model Coefficients",
            "Modeling Competitive Marketing Strategies: The Impact of Marketing-Mix Relationships and Industry Structure",
            "MARKETING MIX MODELING FOR PHARMACEUTICAL COMPANIES ON THE BASIS OF DATA SCIENCE TECHNOLOGIES",
            "ON OPTIMAL BUDGET ALLOCATION USING MARKETING MIX MODELS",
            "A game theory approach to selecting marketing-mix strategies",
            "How Limited Data Access Constrains Marketing-Mix Analytical Efforts",
            "Planning Marketing-Mix Strategies in the Presence of Interaction Effects",
            "MARKETING MIX MODELS USED IN THE LIBERALIZED HOUSEHOLD ELECTRICITY MARKET",
            "Marketing Mix Concept: Blending the Variables to Suit the Contemporary Marketers",
            "The Effects of Marketing Mix Elements on Service Brand Equity",
            "MARKETING MIX IMPLEMENTATION IN SMALL MEDIUM ENTERPRISES: A STUDY OF GALERISTOREY ONLINE BUSINESS",
            "Bayesian Hierarchical Media Mix Model Incorporating Reach and Frequency Data",
            "Media Mix Model Calibration With Bayesian Priors",
            "Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects",
            "Geo-level Bayesian Hierarchical Media Mix Modeling",
            "A Hierarchical Bayesian Approach to Improve Media Mix Models Using Category Data",
            "Bayesian Time Varying Coefficient Model with Applications to Marketing Mix Modeling"
        ]

        for i, paper in enumerate(research_papers):
            if st.button(paper, key=f"research_paper_{i}", use_container_width=True):
                st.session_state.suggested_question = paper
                st.session_state.generate_response = True

        with st.sidebar.expander("Tips to get accurate response ", expanded=False):
            st.write("Try pre-fixing")
            st.write("1. How does Aryma Labs ensure {Your Topic}")
            st.write("2. How Aryma Labs explain {Your Topic}")
            st.write("3. Summarize Aryma Labs approach of {Your topic}")

    # Display the conversation history in reverse order to resemble a chat interface
    chat_container = st.container()
    with chat_container:
        if st.session_state.conversation_history == []:
            col1, col2 = st.columns([1, 8])
            with col1:
                st.image('download.png', width=30)
            with col2:
                st.write("Hello, I am MMM GPT from Aryma Labs. How can I help you?")
                
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
                post_link = ""
            # st.write(r)
            if(language != "en"):
                r = translate(clean_text(r) , "en" , language)
                st.write(r + "\n")
            else:
                st.write(clean_text(r1) + "\n")
            if(post_link != ""):
                if(language != "en"):
                    st.write(translate("For more details, please visit", from_lang='en', to_lang= language) + ": " + post_link)
                else:
                    st.write("For more details, please visit :" + post_link)

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
                    
            st.write("Explore Similiar questions :")
            for i, suggested_question in enumerate(suggested_questions):
                # Ensure unique keys for buttons
                if(suggested_question.page_content != q):
                    if st.button(suggested_question.page_content, key=f"suggested_questions_{idx}_{i}", use_container_width=True):
                        st.session_state.suggested_question = suggested_question.page_content
                        st.session_state.generate_response = True

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
