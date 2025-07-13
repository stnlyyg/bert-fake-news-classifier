#config python and create dir in container
FROM python:3.10-slim-bookworm
WORKDIR /app

# Copy requirements.txt to container
COPY requirements.txt .

# Container run this commands to install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy files from this dir(exclude file in .dockerignore) to /app dir in the container
COPY . .

CMD ["uvicorn", "src.api", "--host", "0.0.0.0", "--port", "80"]