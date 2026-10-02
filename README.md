
# Assignment 1 - FastAPI Word Embedding

This project extends the previous FastAPI application by adding a spaCy word embedding endpoint.

## API Endpoints

### Root

GET /

### Text Generation

POST /generate

Example:

{
  "start_word": "this",
  "length": 10
}

### Word Embedding

POST /embedding

Example:

{
  "word": "apple"
}

## Docker

Build the image:

docker build -t sps-genai .

Run the container:

docker run -p 8000:80 sps-genai

Then open:

http://127.0.0.1:8000/docs
