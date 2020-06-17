FROM python:3.7-slim

### Setup python requirements
COPY requirements.txt .
RUN apt-get update && apt-get install -y g++
RUN pip install --no-cache-dir --no-use-pep517 -r requirements.txt
RUN pip install --no-cache-dir numpy scikit-learn==0.20.3
RUN apt-get autoremove -y g++ && apt-get clean
# Copy the code
COPY main.py .

ENTRYPOINT /bin/bash
