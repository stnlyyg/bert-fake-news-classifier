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
# Install dependencies (run in terminal)
pip install -r requirements.txt
```

---

# Data Processing
Dataset for fine-tuning was obtained from Kaggle [(dataset url)](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification). This is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. The entire dataset is merged from four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political) to prevent over-fitting of classifiers and to provide more text data for better ML training.  
(img df.info())  
Dataset is cleaned without lowercasing every word since bert-base-uncased model will do it. Dataset is split into 80:20 ratio for training purpose and is transformed into format that Hugging Face Trainer API friendly. 

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
```

# Model evaluation
Model evaluation metrics uses confusion matrix and processed evaluation dataset.
```
# To do model evaluation, cd to fake-news-classifier/src and run this in terminal
python evaluation.py
```

#
