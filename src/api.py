from fastapi import FastAPI
from pydantic import BaseModel
from  transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import config

app = FastAPI(
    title="Fake News Classifier API",
    description="API to classify news text as Real or Fake.",
    version="1.0.0"
)

try:
    model = AutoModelForSequenceClassification.from_pretrained(config.HUGGINGFACE_HUB_MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.HUGGINGFACE_HUB_MODEL_PATH)
except Exception as e:
    print(f'Error loading model: {e}')
    exit()

class NewsText(BaseModel):
    text: str

@app.get("/")
def get_root():
    return{"message": "Welcome to the Fake News Classifier API. Use the /docs endpoint to see the documentation."}

@app.get("/classify-news")
def classify_news(news):
    if not news or not news.strip():
        return {}

    inputs = tokenizer(news, return_tensors='pt', truncation=True, padding=True, max_length=config.MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_index = torch.argmax(probs).item()
    confidence = probs[0][label_index]

    predicted_label = config.LABELS[label_index]

    return f'This news is: {predicted_label}, confidence: {confidence}'