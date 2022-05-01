import streamlit as st
import base64
import os
import capstone_project_grp13_eda_dataprep as eda
import cap_multi_input_nlpaug as modelling
import pickle
import random
import numpy as np

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# from zipfile import ZipFile
# import io
from keras.models import load_model

import nltk
#nltk.download('all')

from string import punctuation
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression


data_path = r''
model_name = 'nlpmodel'
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

        st.download_button(label='Process and Download Data Pre Processing File' , data = dataprep, file_name='Capstone_Project_Grp13_EDA_DataPrep.csv', mime="text/csv")

    with col2:

        st.markdown('**NLP Modelling**') 

        # def get_all_file_paths(directory_path):
        #     file_paths = []

        #     # crawling through directory and subdirectories
        #     for root, directories, files in os.walk(directory_path):
        #         for filename in files:
        #             # join the two strings in order to form the full filepath.
        #             filepath = os.path.join(root, filename)
        #             file_paths.append(filepath)
        #     return file_paths

        def processModel():
            with col2:
                # model = modelling.processModel()
                # model.save(data_path + model_name)
                # #directory_path = Path(data_path + model_name)
                # directory_path = (data_path + model_name)
                # file_paths = get_all_file_paths(directory_path)

                # zip_buffer = io.BytesIO()
                # with ZipFile(zip_buffer, 'w') as zip_file:
                #     # writing each file one by one
                #     for file in file_paths:
                #         zip_file.write(file)

                zip_buffer = model_name
                #output_model = pickle.dumps(model)
                #b64 = base64.b64encode(output_model).decode()
                #href = f'<a href="data:file/output_model;base64,{b64}" download="nlpmodel.pkl">Process and Download Trained Model .pkl File</a> (save as .pkl)'
                #st.markdown(href, unsafe_allow_html=True)
                st.download_button("Download Trained Model .zip File", zip_buffer, "nlpmodel.zip")
        
        st.button('Train and Download Model', on_click=processModel) 
    with col3:
        st.markdown('**NLP Industrial Safety Chatbot**') 
        st.button('Industrial Safety NLP Chatbot')

        with col3:

                #form = st.form(key="my_form")
                st.markdown('Welcome to Chatbot Service! Let me know how can I help you')
                input = st.text_input('User:')
                inputadd = st.text_input('Other Details:')

                stop_words = stopwords.words("english")
                @st.cache
                def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
                    # Clean the text, with the option to remove stop_words and to lemmatize word
                
                    # Clean the text
                    text = re.sub(r"[^A-Za-z0-9]", " ", text)
                    text = re.sub(r"\'s", " ", text)
                    text = re.sub(r"http\S+", " link ", text)
                    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
                
                    # Remove punctuation from text
                    text = "".join([c for c in text if c not in punctuation])
                
                    # Optionally, remove stop words
                    if remove_stop_words:
                        text = text.split()
                        text = [w for w in text if not w in stop_words]
                        text = " ".join(text)
                
                    # Optionally, shorten words to their stems
                    if lemmatize_words:
                        text = text.split()
                        lemmatizer = WordNetLemmatizer()
                        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
                        text = " ".join(lemmatized_words)
                
                    # Return a list of words
                    return text

                def make_prediction(review, other):
 
                    # clearn the data
                    clean_review = text_cleaning(review)
                
                    # load the model and make prediction
                    model = load_model(data_path + model_name)
                
                    # make prection
                    other = [ 0,  3,  1,  1,  1, 16]
                    result = model.predict([clean_review, other])
                
                    # check probabilities
                    pred = np.argmax(result[0])
                    potpred = np.argmax(result[1])
                    return pred, potpred

                #submit = st.button(label="Make Prediction")

                if input and inputadd:
                    # pred, potpred = make_prediction(input, 0)
                    # st.write(f'Chatbot: {pred}')
                    pred = 'TEST2'
                    potpred = 'TEST3'
                    st.write(f'Chatbot: Predicted Accident Level - {pred} | Predicted Potential Accident Level - {potpred}')

                ########################### START OF MICROSOFT CODE ########################### 
                # @st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
                # def load_data():    
                #     #model_file = open(data_path + model_name, 'rb')
                #     #loaded_model = base64.b64decode(pickle.load(open(data_path + model_name, 'rb')))
                #     #loaded_model = pickle.loads(base64.b64decode(model_file, "base64"))
                #     #loaded_model = load_model(data_path + model_name)
                #     tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
                #     model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
                #     #model = loaded_model
                #     return tokenizer, model

                # tokenizer, model = load_data()

                # st.markdown('Welcome to Chatbot Service! Let me know how can I help you')
                # input = st.text_input('User:')
                # inputadd = st.text_input('Other Details:')

                # if 'count' not in st.session_state or st.session_state.count == 6:
                #     st.session_state.count = 0 
                #     st.session_state.chat_history_ids = None
                #     st.session_state.old_response = ''
                # else:
                #     st.session_state.count += 1

                # new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

                # bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

                # st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

                # response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

                # if st.session_state.old_response == response:
                #     bot_input_ids = new_user_input_ids
 
                # st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
                # response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

                # st.write(f'Chatbot: {response}')

                # st.session_state.old_response = response
                ########################### END OF MICROSOFT CODE ########################### 

        

        


        


