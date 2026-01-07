FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY flask_app/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

COPY nltk_data /usr/local/nltk_data
ENV NLTK_DATA=/usr/local/nltk_data

RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

EXPOSE 5000

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
