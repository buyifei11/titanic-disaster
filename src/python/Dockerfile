FROM python:3.11-slim

WORKDIR /app

# Install pinned Python packages first (cached between builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the script
COPY titanic_model.py .

# The dataset is NOT copied into the image; it is mounted at run time
ENV DATA_DIR=/app/data

CMD ["python", "titanic_model.py"]