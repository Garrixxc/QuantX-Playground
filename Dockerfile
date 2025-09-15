# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# minimal system deps (faster builds)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

# deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# Streamlit env: headless, no CORS/XSRF, base path at /
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLECORS=false \
    STREAMLIT_SERVER_ENABLEXSRFPROTECTION=false \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

# Vercel provides $PORT at runtime. For local docker run we default to 8000.
ENV PORT=8000
EXPOSE 8000

CMD bash -lc 'echo "Starting on PORT=${PORT}"; streamlit run app.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-8000} \
  --server.headless=true'