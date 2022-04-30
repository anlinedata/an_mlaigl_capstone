import streamlit as st
import base64
import os
import capstone_project_grp13_eda_dataprep as eda
import cap_multi_input_nlpaug as modelling
import pickle
import random
import numpy as np

import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import nltk
nltk.download('all')
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
                href = f'<a href="data:file/output_model;base64,{b64}" download="nlpmodel.pkl">Process and Download Trained Model .pkl File</a> (save as .pkl)'
                st.markdown(href, unsafe_allow_html=True)
        
        st.button('Process and Download Trained Model', on_click=processModel) 
    with col3:
        st.markdown('**NLP Industrial Safety Chatbot**') 
        st.button('Start Industrial Safety NLP Chatbot')

        with col3:

                @st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
                def load_data():    
                    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                    return tokenizer, model

                tokenizer, model = load_data()

                st.markdown('Welcome to Chatbot Service! Let me know how can I help you')
                input = st.text_input('User:')

                if 'count' not in st.session_state or st.session_state.count == 6:
                    st.session_state.count = 0 
                    st.session_state.chat_history_ids = None
                    st.session_state.old_response = ''
                else:
                    st.session_state.count += 1

                new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

                bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

                st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

                response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

                if st.session_state.old_response == response:
                    bot_input_ids = new_user_input_ids
 
                st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

                st.write(f'Chatbot: {response}')

                st.session_state.old_response = response

        

        


        


