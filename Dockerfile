# Use a small Python image
FROM python:3.11-slim

# System deps (optional but nice for speed/plotly)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Streamlit runtime config
ENV PORT=8000 \
    STREAMLIT_BROWSER_GATHERUSAGESTATS=false

EXPOSE 8000

# Start Streamlit; bind to 0.0.0.0 and Vercel-provided $PORT
CMD streamlit run app.py \
    --server.headless=true \
    --server.address=0.0.0.0 \
    --server.port=$PORT \
    --browser.gatherUsageStats=false
