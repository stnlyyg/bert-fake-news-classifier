# Set up Trainer API → Train model → Save model
from transformers import TrainingArguments, Trainer 
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import bert_model
from dataset import bert_tokenizer, train_dataset, eval_dataset

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

training_args = TrainingArguments(
    output_dir = "model/fake-news-detection",
    seed = 42,
    fp16 = True,
    eval_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 2,
    per_device_eval_batch_size = 4,
    num_train_epochs = 5,
    weight_decay = 0.01,
    logging_dir = "./logs",
    load_best_model_at_end = True,
    metric_for_best_model = "f1",
    save_total_limit = 1
)

trainer = Trainer(
    model = bert_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class = bert_tokenizer,
    compute_metrics = compute_metrics
)

trainer.train()