import streamlit as st
import base64
import os
import capstone_project_grp13_eda_dataprep as eda
import cap_multi_input_nlpaug as modelling
import pickle
import random
import numpy as np

import nltk
nltk.download()
#from streamlit_chat import message as st_message

# from transformers import BlenderbotTokenizer
# from transformers import BlenderbotForConditionalGeneration



data_path = r''
file_name = 'IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv'
output_file_name = 'capproj1.csv'

st.set_page_config('Home') 
header = st.container()
body = st.container()
# features = st.container()
# model_training = st.container()

with header:
    st.header('Welcome to EDA Report, NLP Modelling and Industrial Safety NLP Chatbot.')
    st.subheader('Team - Capstone Group 13 - AIML May 2022')
    st.markdown('* Team Members : Kuldeep Sengar | Shushil Anand | Ragunathan Ravichandran | Divya Nyalakonda | Aditya Naik')
    st.markdown('')

with body:
    st.subheader('Please select the below actions')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('**EDA and Data Pre Processing**') 

        def processEDA():
            data = eda.processEDAData()
            return data.to_csv()
        
        dataprep = processEDA()

        with open('Capstone_Project_Grp13_EDA_DataPrep.pdf', 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
            st.download_button(label='Download EDA Report and Data Pre Processing', data = PDFbyte, file_name='Capstone_Project_Grp13_EDA_DataPrep.pdf', mime='application/octet-stream')

        # st.button(label='Process and Download EDA Report and Data Pre Processing' , on_click= processEDA)

        st.download_button(label='Download Data Pre Processing File' , data = dataprep, file_name='Capstone_Project_Grp13_EDA_DataPrep.csv', mime="text/csv")

    with col2:

        st.markdown('**NLP Modelling**') 
        def processModel():
            with col2:
                model = modelling.processModel()
                output_model = pickle.dumps(model)
                b64 = base64.b64encode(output_model).decode()
                href = f'<a href="data:file/output_model;base64,{b64}">Process and Download Trained Model .pkl File</a> (save as .pkl)'
                st.markdown(href, unsafe_allow_html=True)
        
        st.button('Process and Download Trained Model', on_click=processModel) 
    with col3:
        st.markdown('**NLP Industrial Safety Chatbot**') 
        # def get_models():
        #     # it may be necessary for other frameworks to cache the model
        #     # seems pytorch keeps an internal state of the conversation
        #     model_name = "facebook/blenderbot-400M-distill"
        #     tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
        #     model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
        #     return tokenizer, model

        if "history" not in st.session_state:
            st.session_state.history = []

        def generate_answer():
            # tokenizer, model = get_models()
            user_message = st.session_state.input_text
            # inputs = tokenizer(st.session_state.input_text, return_tensors="pt")
            # result = model.generate(**inputs)

            # message_bot = tokenizer.decode(
            #     result[0], skip_special_tokens=True   ) 
            message_bot = "TEST"
          

            st.session_state.history.append({"message": user_message, "is_user": True})
            st.session_state.history.append({"message": message_bot, "is_user": False})

        def get_bot_response(message):
            message = message.lower()
            response = "TEST"
            # results = model.predict([bag_of_words(message,words)])[0]
            # result_index = np.argmax(results)
            # tag = labels[result_index]
            # for tg in data['intents']:
            #     if tg['tag'] == tag:
            #     responses = tg['responses']
            # response = random.choice(responses)
            return str(response)

        #st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)
        def get_text():
            input_text = st.text_input("You: ","So, what's in your mind")
            return input_text

        def generate_answer1():
            user_message = st.session_state.input_text
            if user_message=="Bye" or user_message=='bye':
                    st.markdown('ChatBot: Bye')
                    #break
            else:
                    st.markdown('ChatBot: AAA')
                    response=get_bot_response(user_message)
                    st.markdown('ChatBot: BBB')
                    st.markdown('ChatBot: AAA')
                    st.text_area('ChatBot: BBB')

        def startchatbot():
            #st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)

            #name = input('Enter Your Name: ')
            st.text_input('Enter Your Name:', key="input_text", on_change=generate_answer1)
            st.markdown('Welcome to Chatbot Service! Let me know how can I help you')
            # response = ""

            # if True:

            #     request = st.text_input(':')

            #     if request=="Bye" or request=='bye':
            #         st.markdown('ChatBot: Bye')
            #         #break
            #     else:
            #         st.markdown('ChatBot: AAA')
            #         response=get_bot_response(request)
            #         st.markdown('ChatBot: BBB')
            #         st.markdown('ChatBot: AAA')
            #         st.text_area('ChatBot: BBB')
        # for chat in st.session_state.history:
        #     st_message(**chat)  # unpacking

        st.button('Start Industrial Safety NLP Chatbot', on_click= startchatbot)

        


        


