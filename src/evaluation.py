# Load trained model → Predict on test set → Evaluate (confusion matrix, F1, etc)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import eval_dataset
import config

def evaluate_model():
    this_model = AutoModelForSequenceClassification.from_pretrained(config.BASE_MODEL_PATH)

    eval_args = TrainingArguments(**config.EVAL_ARGS)

    trainer = Trainer(
        model = this_model,
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