FROM python:3.11-slim

WORKDIR /app

# Faster build & smaller image
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8000
EXPOSE 8000

# Use shell so ${PORT} expands on Vercel; fallback to 8000 for local docker run
CMD bash -lc 'streamlit run app.py \
  --server.headless=true \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-8000} \
  --browser.gatherUsageStats=false'
