FROM python:3.12-slim
 
WORKDIR /app
 
COPY requirements.txt .
 
RUN pip install --no-cache-dir -r requirements.txt
 
COPY common.py .
COPY ingest.py .
COPY server.py .
COPY urls.txt .
 
CMD ["python", "server.py"]