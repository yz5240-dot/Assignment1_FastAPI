
from fastapi import FastAPI
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding


app = FastAPI(
    title="Assignment 1 FastAPI",
    description="Text generation and word embedding API",
    version="1.0.0"
)


corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]


bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {
        "message": "FastAPI application is running"
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )

    return {
        "generated_text": generated_text
    }


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    embedding = calculate_embedding(request.word)

    return {
        "word": request.word,
        "embedding": embedding
    }
