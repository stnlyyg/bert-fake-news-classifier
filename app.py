# Create simple Gradio app → Load model → Predict news text input
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import gradio as gr

model_path = './model/fake-news-detection/checkpoint-10731'

finetuned_model = AutoModelForSequenceClassification.from_pretrained(model_path)
finetuned_tokenizer = AutoTokenizer.from_pretrained(model_path)

labels = ['Real', 'Fake']

def classify_news(text: str):
    if not text or not text.strip():
        return f'Please enter some news text to classify.'

    inputs = finetuned_tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256)

    with torch.no_grad():
        outputs = finetuned_model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_index = torch.argmax(probs).item()
    confidence = probs[0][label_index]

    predicted_label = labels[label_index]

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