import streamlit as st
import base64

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
        with open('Capstone_Project_Grp13_EDA_DataPrep.pdf', 'rb') as pdf_file:
            PDFbyte = pdf_file.read()
            st.download_button(label='Download EDA Report and Data Pre Processing', data = PDFbyte, file_name='Capstone_Project_Grp13_EDA_DataPrep.pdf', mime='application/octet-stream')
        st.button('Process and Download EDA Report and Data Pre Processing')
    with col2:
        st.button('NLP Modelling and Review') 
    with col3:
        st.button('Industrial Safety NLP Chatbot')
    
    


