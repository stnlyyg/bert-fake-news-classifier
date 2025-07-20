from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

#Directories
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODEL_DIR = ROOT_DIR / "model"
REPORTS_DIR = ROOT_DIR / "reports"
SRC_DIR = ROOT_DIR / "src"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

#Data configuration and processing
DATA_FILE = DATA_DIR / "fakenewsdata.csv"
MAX_LENGTH = 256

#Data train test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

PROCESSED_DATA = DATA_DIR / "processed"
PROCESSED_DATA_TRAIN = PROCESSED_DATA / "train"
PROCESSED_DATA_TEST = PROCESSED_DATA / "test"

PROCESSED_DATA.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_TRAIN.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_TEST.mkdir(parents=True, exist_ok=True)

#Model and tokenizer configuration
BASE_MODEL_PATH = "google-bert/bert-base-uncased"
SAVED_MODEL_PATH = MODEL_DIR / "fake-news-detection"
BEST_CHECKPOINT_PATH = SAVED_MODEL_PATH / "checkpoint-10731"
HUGGINGFACE_HUB_MODEL_PATH = "stnleyyg/fake-news-classifier"

#Training 
TRAINING_ARGS = {
    "output_dir": str(SAVED_MODEL_PATH),
    "seed": 42,
    "fp16": True,

    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.1,
    "warmup_ratio": 0.5,

    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "per_device_eval_batch_size": 8,

    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "save_total_limit": 1,

    "logging_dir": LOGS_DIR,
    "logging_strategy": "steps",
    "logging_steps": 100,
    "report_to": "tensorboard"
}

#Evaluation
EVAL_RESULT_DIR = REPORTS_DIR / "eval_results"
EVAL_RESULT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_ARGS = {
    "output_dir": EVAL_RESULT_DIR,
    "per_device_eval_batch_size": 8,
    "do_train": False,
    "do_eval": True
}

#App
LABELS = ["real", "fake"]