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

    num_train_epochs = 3,
    learning_rate = 2e-5,
    weight_decay = 0.1,
    warmup_ratio = 0.5,

    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 2,
    per_device_eval_batch_size = 8,

    eval_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "f1",
    save_total_limit = 1,

    logging_dir = "./logs",
    logging_strategy = "steps",
    logging_steps = 100,
    report_to = "tensorboard"
)

trainer = Trainer(
    model = bert_model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    processing_class = bert_tokenizer,
    compute_metrics = compute_metrics
)

if __name__ == '__main__':
    trainer.train()