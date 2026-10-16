# Assignment 2 - CNN FastAPI

Updated from Assignment 1. Existing `GET /`, `POST /generate` and `POST /embedding` remain available; new `POST /predict` predicts a CIFAR-10 image class.

## 1. Train the CNN (Google Colab recommended)

Upload/extract this project in Colab and enter its directory. Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python train.py --epochs 10 --batch-size 128
```

The training script downloads CIFAR-10, resizes it to 64×64, trains the **specified architecture**, evaluates on the test set, prints actual accuracy, and saves `models/cnn_cifar10.pth`. Accuracy will depend on the run; none is claimed in advance.

If training in Colab, **download the saved .pth file** and place it inside this project's `models/` directory on your own computer **before building Docker**. Do not upload a model checkpoint bigger than your GitHub repository allows; Git LFS is an option.

## 2. Run with Docker

Have Docker Desktop running, from the project directory with `models/cnn_cifar10.pth` present:

```bash
docker build -t assignment2-cnn .
docker run --rm -p 8000:80 assignment2-cnn
```

Open http://127.0.0.1:8000/docs and try `POST /predict` using a JPG/PNG file. Example using curl:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@cat.png"
```

Example response format (not a real prediction):

```json
{"class": "cat", "class_index": 3, "confidence": 0.7532}
```

The Docker image retains the original large spaCy language model, and adds PyTorch, so building can take time and disk space. `/predict` returns HTTP 503 if the trained model file is missing. A missing checkpoint must be fixed, not treated as a completed deployment.

## 3. Original endpoints

- `GET /`
- `POST /generate` body: `{"start_word":"this","length":10}`
- `POST /embedding` body: `{"word":"apple"}`

## 4. Submit

- Push changed code and trained model (or provide teacher a way to obtain it) to GitHub.
- Verify a fresh Docker build and an actual upload with `/predict` returning HTTP 200.
- Answer the six questions in `THEORY_ANSWERS.md` with calculation steps.
- Add screenshots of training logs, Docker start, `/docs` and `/predict` response to your report if useful.

**Important:** This repository contains the implementation, but you must still **run training, add the checkpoint and test the Docker deployment**. We have not claimed those steps have been completed.
