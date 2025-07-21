# Introduction
This is my personal project in order to implement my understanding in my AI Engineering study on using transformers model for NLP task. This is a machine learning project that classifies news article as **real** or **fake**, using a fine-tuned bert-base-uncased transformer model. This project features:

- NLP model with Hugging Face Transformers
- FastAPI backend for prediction service
- Gradio frontend for quick interaction
- Dockerized setup for easy deployment

---

# Project Overview
Fake news has become a major concern across digital media. This project aims to detect and classify fake news using a transformer-based NLP model (like BERT).
This project interaction can be made via:
- A FastAPI REST API
- A Gradio web interface

---

# Tech Stack
| Component   | Technology                          |
|-------------|-------------------------------------|
| NLP Model   | BERT or DistilBERT (Hugging Face)   |
| Framework   | PyTorch, Transformers               |
| Backend     | FastAPI                             |
| Frontend    | Gradio                              |
| Container   | Docker, Docker Compose              |

---

# Setup
```
# Git clone this repo
https://github.com/stnlyyg/fake-news-classifier.git

# Install dependencies (run in root directory terminal)
pip install -r requirements.txt
```

---

# Data Processing
Dataset for fine-tuning was obtained from Kaggle [(dataset url)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). This is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. The entire dataset is merged from four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting of classifiers and to provide more text data for better ML training.  
<img width="1919" height="908" alt="image" src="https://github.com/user-attachments/assets/3eae7664-0374-480d-9bbe-8c89e1d5d88a" />
Training process uses 2 classes, text and label. Title and text column is merged and named text, while label remain as is.  
Dataset is cleaned without lowercasing every word since bert-base-uncased model will do it and is split into 80:20 ratio for training purpose.  
Dataset later transformed into format that Hugging Face Trainer API friendly.

```
# To do data processing, run this in terminal
python data_processing.py
```

---

# Training
Fine-tuning model is done by leveraging Hugging Face Trainer API. Processed dataset will be use as input for Trainer API and training result is saved in a prepared folder. Fine-tuning uses bert-base-uncased pretrained model from google [(pretrained model)](google-bert/bert-base-uncased).
```
# To do training, cd to fake-news-classifier/src and run this in terminal
python train.py

Training result for 3 epochs
{'eval_loss': 0.07674776762723923, 'eval_accuracy': 0.9780542353927872, 'eval_precision': 0.9745983797885487, 'eval_recall': 0.9821502698215027, 'eval_f1': 0.9783597518952447, 'eval_runtime': 90.4567, 'eval_samples_per_second': 158.175, 'eval_steps_per_second': 19.777, 'epoch': 1.0}

{'eval_loss': 0.06746303290128708, 'eval_accuracy': 0.9838551859099804, 'eval_precision': 0.9757889009793254, 'eval_recall': 0.9926663899266639, 'eval_f1': 0.9841552918581521, 'eval_runtime': 85.0274, 'eval_samples_per_second': 168.275, 'eval_steps_per_second': 21.04, 'epoch': 2.0}

{'eval_loss': 0.06384191662073135, 'eval_accuracy': 0.9870701705339671, 'eval_precision': 0.9845857418111753, 'eval_recall': 0.98989898989899, 'eval_f1': 0.987235217001311, 'eval_runtime': 85.1316, 'eval_samples_per_second': 168.069, 'eval_steps_per_second': 21.015, 'epoch': 3.0}
```

# Model evaluation
Model evaluation metrics uses confusion matrix and processed evaluation dataset.
```
# To do model evaluation, cd to fake-news-classifier/src and run this in terminal
python evaluation.py
```
Evaluation result.  
<img width="676" height="509" alt="image" src="https://github.com/user-attachments/assets/bd401170-bafb-41b0-ba01-ce4b4221de21" />

---

# Running the App
The app is separated into backend that contain the API and frontend with gradio for quick demo. There are two ways to run this, via local or docker compose.

## Run Locally
To run the app locally, you must first run the backend in your terminal, then follow by running frontend in separated terminal.
```
# cd to fake-news-classifier/src/backend-api/ and run this in your first terminal
uvicorn api:app --reload
```
<img width="1500" height="257" alt="image" src="https://github.com/user-attachments/assets/2efc12ed-9369-46f2-84cb-d8ecc2dc06c6" />

```
# cd to fake-news-classifier/src/frontend-gradio/ and run this in your second terminal
python gradio_app.py
```
<img width="1501" height="253" alt="image" src="https://github.com/user-attachments/assets/d61cc174-889d-479c-9bf4-789d9bc9c0b8" />  
<br/><br/>
You can open localhost:8000/docs on your browser to use the app through FastAPI UI
<img width="1918" height="1018" alt="image" src="https://github.com/user-attachments/assets/31d4343a-1a59-4905-9997-503de9f75b0c" />  
<br/><br/>
You can open localhost:7860 on your browser to use the gradio app via connection to the backend
<img width="1918" height="1012" alt="image" src="https://github.com/user-attachments/assets/0616307e-9e02-4b48-8388-72a0ea537ffd" />

---

## Run with Docker Compose
To run the app via Docker, make sure your docker system is up and running before proceeding to next step.
```
# In root directory, run this in your terminal
docker-compose up --build

You can open localhost:8000/docs on your browser to use the app through FastAPI UI  
You can open localhost:7860 on your browser to use the gradio app via connection to the backend
```
