from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src import config

model = AutoModelForSequenceClassification.from_pretrained(config.HUGGINGFACE_HUB_MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(config.HUGGINGFACE_HUB_MODEL_PATH)

def classify_news(text: str):
    if not text or not text.strip():
        return {}

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=config.MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_index = torch.argmax(probs).item()
    confidence = probs[0][label_index]

    predicted_label = config.LABELS[label_index]

    return f'This news is: {predicted_label}, confidence: {confidence}'

def classifier_app():
    demo_app = gr.Interface(
        fn=classify_news,
        inputs=gr.TextArea(placeholder="Enter any news...", label='Input news: '),
        outputs=gr.TextArea(label="News classified as: "),
        title="Fake News Classifier"
    )

    demo_app.launch()

if __name__ == '__main__':
    classifier_app()