
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding
from app.inference import ImageClassifier


app = FastAPI(
    title="Assignment 2 CNN FastAPI",
    description="Text generation, word embeddings, and CIFAR-10 image classification",
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


image_classifier = ImageClassifier()


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not image_classifier.available:
        raise HTTPException(status_code=503, detail="Model not trained yet. Run train.py and save models/cnn_cifar10.pth")
    if file.content_type is not None and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image too large (max 10MB)")
    try:
        return image_classifier.predict(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
