FROM python:3.7
WORKDIR /app
COPY requirements.txt . /app
RUN pip install -r requirements.txt
EXPOSE 8501
COPY ./app /app
ENTRYPOINT ["streamlit", "run"]

CMD ["main.py"]