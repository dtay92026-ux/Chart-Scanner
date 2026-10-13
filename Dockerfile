FROM python:3.12-slim

# Tesseract is the OCR engine that reads the price and time labels on the chart.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render sets PORT. One worker, because the result cache lives in memory.
CMD gunicorn -w 1 --threads 4 -t 150 -b 0.0.0.0:${PORT:-5000} app:app
