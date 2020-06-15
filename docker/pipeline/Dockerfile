FROM python:3.8-slim

### Setup python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --no-use-pep517 -r requirements.txt

# Copy the code
COPY main.py .

ENTRYPOINT /bin/bash
