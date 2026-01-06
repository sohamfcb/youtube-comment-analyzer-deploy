FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1

COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

COPY nltk_data /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

RUN pip install --default-timeout=300 -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]
