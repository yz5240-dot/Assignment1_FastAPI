
FROM python:3.12-slim-bookworm

RUN pip install --no-cache-dir fastapi uvicorn spacy

RUN python -m spacy download en_core_web_lg

WORKDIR /code

COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
