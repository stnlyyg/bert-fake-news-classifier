# Load tokenizer and model (bert-base-uncased)
from transformers import AutoModelForSequenceClassification

bert_model = AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=2)