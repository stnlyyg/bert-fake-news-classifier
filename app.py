# Create simple Gradio app → Load model → Predict news text input
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = './model/fake-news-detection/checkpoint-10731'

app_model = AutoModelForSequenceClassification.from_pretrained(model_path)
app_tokenizer = AutoTokenizer.from_pretrained(model_path)