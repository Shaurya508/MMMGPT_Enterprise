# from io import BytesIO
# from PIL import Image , UnidentifiedImageError
# import requests
# import streamlit as st
# # import streamlit as st
# import pyttsx3
# # import threading
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# # for voice in voices:
# #     engine.setProperty('voice', voice.id)
# #     print(voice.id)
# #     engine.say('The quick brown fox jumped over the lazy dog.')
# #     engine.runAndWait()# # Initialize the TTS engine
# def onWord(name, location, length):
#     print ('word', name, location, length)
#     if location > 1:
#         engine.stop()
# engine = pyttsx3.init()
# engine.connect('started-word', onWord)
# engine.say('The quick brown fox jumped over the lazy dog.')
# engine.runAndWait()

# # engine = pyttsx3.init()
# # engine_lock = threading.Lock()  # To ensure thread safety

# # # Function to handle speaking in a thread
# # def speak(text, stop_event):
# #     if engine._inLoop:
# #         engine.endLoop()
# #     with engine_lock:
# #         # Queue up the text to speak
# #         engine.say(text)
# #         while not stop_event.is_set():
# #             engine.runAndWait()  # Process the speech
# #             if stop_event.is_set():
# #                 engine.stop()  # Stop speaking if stop_event is set

# # # Streamlit app
# # st.title("Text to Speech with Stop Button")

# # # Input and control buttons
# # text_input = st.text_area("Enter text to speak:")
# # start_button = st.button("Start Speaking")
# # stop_button = st.button("Stop Speaking")

# # # Create an event to control stopping
# # stop_event = threading.Event()

# # # Start speaking when button is pressed
# # if start_button and text_input:
# #     stop_event.clear()  # Clear the stop event before starting
# #     # Start speech in a new thread
# #     threading.Thread(target=speak, args=(text_input, stop_event), daemon=True).start()

# # # Stop speaking when button is pressed
# # if stop_button:
# #     stop_event.set()  # Set the event to stop speaking
# # response = requests.get('https://media.licdn.com/dms/image/C5622AQGqW-i55CLi-w/feedshare-shrink_800/0/1670415342997?e=2147483647&v=beta&t=3C_2SId-uQyO6I6KdaDe8urhbzPdXibnueSJqeWIsfE')
# # img = Image.open(BytesIO(response.content))
# # st.image(img, use_column_width=True)
import argostranslate.package
import argostranslate.translate



# Download and install Argos Translate package
# argostranslate.package.update_package_index()
# available_packages = argostranslate.package.get_available_packages()
# package_to_install = next(
#     filter(
#         lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
#     )
# )
# argostranslate.package.install_from_path(package_to_install.download())

# Translate
def translate(r , from_lang , to_lang):
    from_code = from_lang
    to_code = to_lang
    # Download and install Argos Translate package
    argostranslate.package.update_package_index()
    available_packages = argostranslate.package.get_available_packages()
    package_to_install = next(
        filter(
            lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
        )
    )
    argostranslate.package.install_from_path(package_to_install.download())
    translatedText = argostranslate.translate.translate(r, from_code, to_code)
    return translatedText