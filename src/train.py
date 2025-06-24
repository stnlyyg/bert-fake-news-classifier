# Set up Trainer API → Train model → Save model
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config
from data_processing import bert_tokenizer, train_dataset, eval_dataset

this_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_PATH)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(**config.TRAINING_ARGS)

trainer = Trainer(
    model = this_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class = bert_tokenizer,
    compute_metrics = compute_metrics
)

if __name__ == '__main__':
    trainer.train()