# Load dataset from Kaggle → Preprocess → Split into train/test
import pandas as pd
import re
import config

from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer

df = pd.read_csv(config.DATA_FILE)

df.dropna(inplace=True)
df['text'] = df['title'] + ". " + df['text']
df = df.drop(columns=['Unnamed: 0', 'title'], axis=1)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip()
    return text

df['text'] = df['text'].apply(clean_text)
df.reset_index(drop=True, inplace=True)

train_texts, eval_texts, train_labels, eval_labels = train_test_split(df['text'].tolist(), df['label'].tolist(), test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

train_labels = [int(label) for label in train_labels]
eval_labels = [int(label) for label in eval_labels]

bert_tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_PATH)

def tokenize_function(texts):
    return bert_tokenizer(texts, truncation=True, padding=True, max_length=config.MAX_LENGTH)

train_encodings = tokenize_function(train_texts)
eval_encodings = tokenize_function(eval_texts)

train_dataset = Dataset.from_dict({'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'label': train_labels})
eval_dataset = Dataset.from_dict({'input_ids': eval_encodings['input_ids'], 'attention_mask': eval_encodings['attention_mask'], 'label': eval_labels})

train_dataset.save_to_disk(config.PROCESSED_DATA_TRAIN)
eval_dataset.save_to_disk(config.PROCESSED_DATA_TEST)