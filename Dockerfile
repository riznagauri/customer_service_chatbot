FROM python:3.6
ADD . /chatbot-v3
WORKDIR /chatbot-v3
RUN pip install -r requirements.txt -f  https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m spacy download en
EXPOSE 8080
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "app:app"]
