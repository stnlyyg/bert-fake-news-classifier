import gradio as gr
import requests
import os

USER_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"

if USER_DOCKER:
    API_URL = "http://backend:80/classify_news/"
else:
    API_URL = "http://127.0.0.1:8000/classify_news/"

def classify_news(text: str):
    if not text or not text.strip():
        return f"News can't be empty or whitespace"

    payload = {"news": text}
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()

    result = response.json().get("model_result", "Classification failed.")
    
    return result


demo_app = gr.Interface(
    fn=classify_news,
    inputs=gr.TextArea(placeholder="Enter any news...", label='Input news: '),
    outputs=gr.TextArea(label="News classified as: "),
    title="Fake News Classifier"
)

demo_app.launch(server_name="0.0.0.0")