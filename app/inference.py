"""Load trained CIFAR-10 CNN and predict on uploaded RGB images."""
import io
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.cnn_model import CIFAR10_CLASSES, CNNClassifier

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cnn_cifar10.pth"
preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3),
])


class ImageClassifier:
    def __init__(self, checkpoint=MODEL_PATH):
        self.model = CNNClassifier()
        self.available = Path(checkpoint).is_file()
        if self.available:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.model.eval()

    def predict(self, raw_bytes: bytes):
        try:
            with Image.open(io.BytesIO(raw_bytes)) as original:
                img = original.convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ValueError("Uploaded content is not a supported image") from exc
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0]
        index = int(probabilities.argmax().item())
        return {"class": CIFAR10_CLASSES[index], "class_index": index,
                "confidence": round(float(probabilities[index].item()), 4)}
