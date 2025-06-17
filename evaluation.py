# Load trained model → Predict on test set → Evaluate (confusion matrix, F1, etc)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix
from dataset import eval_dataset

model_path = "model/fake-news-detection/checkpoint-10731"

def evaluate_model():
    bert_model = AutoModelForSequenceClassification.from_pretrained(model_path)

    eval_args = TrainingArguments(
        output_dir = "./eval_results",
        per_device_eval_batch_size = 8,
        do_train = False,
        do_eval = True
    )

    trainer = Trainer(
        model = bert_model,
        args = eval_args
    )

    predictions = trainer.predict(eval_dataset) #output of this code are PredictionOutput object containing predictions, label_ids, and metrics
    y_true = predictions.label_ids
    y_pred = np.argmax(predictions.predictions, axis=-1)
    target_names = ["Real", "Fake"]
    eval_report = classification_report(y_true, y_pred, target_names=target_names, digits=4)

    print(f'Classification evaluation report: \n{eval_report}')

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')

    plt.show()

if __name__ == '__main__':
    evaluate_model()