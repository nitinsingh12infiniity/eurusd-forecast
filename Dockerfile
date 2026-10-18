FROM python:3.14-slim
 
WORKDIR /app
 
# Install the small CPU-only version of PyTorch first (the normal one is several GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY . .
 
ENV PORT=5001
CMD gunicorn --workers 1 --threads 4 --timeout 120 --bind 0.0.0.0:${PORT} app:app
