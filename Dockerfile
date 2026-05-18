FROM python:3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt_tab')"

COPY . .

EXPOSE 7860

CMD ["python", "main.py", "-t", "webrtc", "--host", "0.0.0.0", "--port", "7860"]