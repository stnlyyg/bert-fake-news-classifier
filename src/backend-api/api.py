from fastapi import FastAPI
from pydantic import BaseModel
from  transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

app = FastAPI(
    title="Fake News Classifier API",
    description="API to classify news text as Real or Fake.",
    version="1.0.0"
)

try:
    model = AutoModelForSequenceClassification.from_pretrained(config.BEST_CHECKPOINT_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.BEST_CHECKPOINT_PATH)
except Exception as e:
    print(f'Error loading model: {e}')
    exit()

@app.get("/")
def get_root():
    return{"message": "Welcome to the Fake News Classifier API. Use the /docs endpoint to see the documentation."}

class NewsText(BaseModel):
    news: str

@app.post("/classify_news/")
def classify_news(news: NewsText):
    news_text = news.news

    if not news_text or not news_text.strip():
        return f"Error: Cannot input blank or whitespace."

    inputs = tokenizer(news.news, return_tensors='pt', truncation=True, padding=True, max_length=config.MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_index = torch.argmax(probs).item()
    confidence = probs[0][label_index]

    predicted_label = config.LABELS[label_index]

    return {"model_result": f"This news is predicted as {predicted_label} with confidence of {confidence}"}