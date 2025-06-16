# Load dataset from Kaggle → Preprocess → Split into train/test
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

df_addfake = pd.read_csv('../fakenewsdata.csv')
df_addfake.dropna(inplace=True)

df_addfake['text'] = df_addfake['title'] + " " + df_addfake['text']
df_addfake = df_addfake.drop(['Unnamed: 0', 'title'], axis=1)

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)               # Remove HTML tags
    text = re.sub(r"http\S+", "", text)             # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)         # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()        # Remove extra spaces
    return text

df_addfake['text'] = df_addfake['text'].apply(clean_text)

train_texts, eval_texts, train_labels, eval_labels = train_test_split(df_addfake['text'].tolist(), df_addfake['label'].tolist(), test_size=0.2, random_state=42, stratify=df_addfake['label'].tolist())

bert_tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-uncased')

def tokenize_function(texts):
    return bert_tokenizer(texts, truncation=True, padding=True, max_length=512)

train_encodings = tokenize_function(train_texts)
eval_encodings = tokenize_function(eval_texts)

train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'label': train_labels})
eval_dataset = Dataset.from_dict({'input_ids': eval_encodings['input_ids'], 'attention_mask': eval_encodings['attention_mask'], 'label': eval_labels})