# ==============================================================================
# Import Necessary Libraries
# ==============================================================================
import os                       # For interacting with the operating system (paths, directories)
import re                       # For regular expressions (used in text cleaning)
import pandas as pd             # For data manipulation (loading data into DataFrames)
import numpy as np              # For numerical operations (used in metrics calculation)
import matplotlib.pyplot as plt # For creating plots (like the confusion matrix)
import seaborn as sns           # For enhancing plot aesthetics (confusion matrix heatmap)
import torch                    # PyTorch library (check for GPU)
import evaluate                 # Hugging Face library for evaluation metrics
import joblib                   # Potentially needed if saving differently, though Trainer handles it
from datasets import Dataset, DatasetDict # Hugging Face library for handling datasets
from transformers import (
    AutoTokenizer,                  # Automatically loads the correct tokenizer for a model
    AutoModelForSequenceClassification, # Automatically loads a model fine-tunable for sequence classification
    TrainingArguments,              # Class to configure training parameters
    Trainer                         # Class that handles the training and evaluation loop
)
from sklearn.model_selection import train_test_split # For stratified sampling
from sklearn.metrics import confusion_matrix         # For creating the confusion matrix

print("--- DistilBERT Fine-Tuning Script ---")

# ==============================================================================
# Configuration Section
# ==============================================================================
# --- File Paths (Assumes running from project/ directory) ---
DATA_DIR = os.path.join('data', 'aclImdb')                          # Path to the main dataset directory
RESULTS_DIR = os.path.join('results')                               # Main directory for results
MODELS_DIR = os.path.join('models')                                 # Main directory for saved models

# --- Subdirectories for this specific model ---
DISTILBERT_RESULTS_SUBDIR = os.path.join(RESULTS_DIR, 'distilbert') # Subdir for DistilBERT evaluation results
DISTILBERT_MODEL_SAVE_DIR = os.path.join(MODELS_DIR, 'distilbert_fine_tuned') # Final fine-tuned model saved here
DISTILBERT_TRAINING_OUTPUT_DIR = os.path.join(RESULTS_DIR, 'distilbert_training_output') # Checkpoints and logs during training

# --- Dataset Paths ---
TRAIN_DIR = os.path.join(DATA_DIR, 'train')                         # Path to training data
TEST_DIR = os.path.join(DATA_DIR, 'test')                           # Path to test data

# --- Model Identifier ---
# Specifies which pre-trained model to load from Hugging Face Hub
MODEL_NAME = "distilbert-base-uncased"

# --- Data Sampling Configuration ---
# Set sample sizes to reduce data for faster training/evaluation (especially on CPU)
# Set to None to use the full dataset for that split.
# Using smaller samples will be much faster but might yield less robust results.
TRAIN_SAMPLE_SIZE = 100  # Total number of training samples (e.g., 2000 positive, 2000 negative)
TEST_SAMPLE_SIZE = 50   # Total number of test samples for evaluation (e.g., 500 positive, 500 negative)

# --- Training Hyperparameters ---
# These parameters control the fine-tuning process. Adjust them based on performance and resources.
NUM_EPOCHS = 3 # Number of times to iterate over the training dataset (start low, e.g., 1-3)
LEARNING_RATE = 2e-5  # Controls how much the model weights are adjusted (common value for fine-tuning)
TRAIN_BATCH_SIZE = 8  # Number of samples processed in one batch during training (decrease if GPU memory is low)
EVAL_BATCH_SIZE = 8  # Number of samples processed in one batch during evaluation (decrease if GPU memory is low)
WEIGHT_DECAY = 0.01 # Regularization technique to prevent overfitting

# ==============================================================================
# Setup: Directory Creation and GPU Check
# ==============================================================================
# --- Ensure required directories exist ---
os.makedirs(DISTILBERT_RESULTS_SUBDIR, exist_ok=True)
os.makedirs(DISTILBERT_MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(DISTILBERT_TRAINING_OUTPUT_DIR, exist_ok=True) # Trainer needs this output dir

print(f"Fine-tuned model will be saved in: {DISTILBERT_MODEL_SAVE_DIR}")
print(f"Training outputs (checkpoints) will be in: {DISTILBERT_TRAINING_OUTPUT_DIR}")
print(f"Evaluation results will be saved in: {DISTILBERT_RESULTS_SUBDIR}")

# --- Check for GPU availability ---
if torch.cuda.is_available():
    print("\nGPU detected. Using CUDA.")
    device = torch.device("cuda")
else:
    print("\nNo GPU detected. Using CPU (Training will be very slow).")
    device = torch.device("cpu")

# ==============================================================================
# Data Loading and Preparation
# ==============================================================================

# --- Function to load data from structured IMDB folders into a DataFrame ---
def load_imdb_data_to_df(data_dir):
    """Loads IMDB review text files and labels into a pandas DataFrame."""
    texts = []
    labels = []
    # Define paths for positive and negative reviews
    pos_dir = os.path.join(data_dir, 'pos')
    neg_dir = os.path.join(data_dir, 'neg')

    # Check if directories exist
    if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
         raise FileNotFoundError(f"Could not find 'pos' or 'neg' directories in {data_dir}")

    # Load positive reviews (label 1)
    print(f"  Loading positive reviews from: {pos_dir}")
    for filename in os.listdir(pos_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(pos_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1) # 1 for positive
            except Exception as e:
                print(f"  Warning: Could not read file {filepath}: {e}")

    # Load negative reviews (label 0)
    print(f"  Loading negative reviews from: {neg_dir}")
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(neg_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0) # 0 for negative
            except Exception as e:
                print(f"  Warning: Could not read file {filepath}: {e}")

    # Create DataFrame
    df = pd.DataFrame({'text': texts, 'label': labels})
    # Basic cleaning: Remove HTML tags (like <br />)
    df['text'] = df['text'].apply(lambda x: re.sub(re.compile('<.*?>'), ' ', x))
    # Shuffle the DataFrame to mix positive and negative samples
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# --- Load FULL Data ---
print("\nLoading FULL dataset initially...")
try:
    full_train_df = load_imdb_data_to_df(TRAIN_DIR)
    full_test_df = load_imdb_data_to_df(TEST_DIR)
    print(f"Loaded {len(full_train_df)} training and {len(full_test_df)} test samples.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the IMDB dataset is correctly placed in project/data/aclImdb/")
    exit()

# --- Perform Stratified Sampling (if configured) ---
sampled_train_df = full_train_df
if TRAIN_SAMPLE_SIZE is not None and TRAIN_SAMPLE_SIZE < len(full_train_df):
    print(f"\nSampling training data down to {TRAIN_SAMPLE_SIZE} stratified samples...")
    # Use train_test_split for stratified sampling based on labels
    # We only need the sampled part, so the second returned value (_) is ignored
    sampled_train_df, _ = train_test_split(
        full_train_df,
        train_size=TRAIN_SAMPLE_SIZE / len(full_train_df), # Specify fraction
        stratify=full_train_df['label'], # Keep positive/negative balance
        random_state=42 # Ensures reproducibility
    )
    print(f"Using {len(sampled_train_df)} training samples.")
else:
    print("\nUsing full training dataset.")

sampled_test_df = full_test_df
if TEST_SAMPLE_SIZE is not None and TEST_SAMPLE_SIZE < len(full_test_df):
    print(f"Sampling test data down to {TEST_SAMPLE_SIZE} stratified samples...")
    sampled_test_df, _ = train_test_split(
        full_test_df,
        train_size=TEST_SAMPLE_SIZE / len(full_test_df), # Specify fraction
        stratify=full_test_df['label'], # Keep positive/negative balance
        random_state=42
    )
    print(f"Using {len(sampled_test_df)} test samples for evaluation.")
else:
    print("Using full test dataset for evaluation.")
    # Note: Evaluating on the full test set gives a more reliable final measure,
    # but sampling speeds up evaluation during development, especially on CPU.

# --- Convert Sampled DataFrames to Hugging Face Dataset objects ---
# The Trainer API works best with the `datasets` library format
print("\nConverting sampled data to Hugging Face Dataset format...")
train_dataset = Dataset.from_pandas(sampled_train_df)
test_dataset = Dataset.from_pandas(sampled_test_df)

# Combine into a DatasetDict, which is convenient for the Trainer
raw_datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset # Using the (potentially sampled) test set for evaluation
})
print("\nSampled Dataset structure:")
print(raw_datasets)

# ==============================================================================
# Preprocessing: Tokenization
# ==============================================================================
# Load the tokenizer specific to the chosen pre-trained model (DistilBERT)
print(f"\nLoading tokenizer for model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the function that will tokenize the text data
def tokenize_function(examples):
    """Applies the tokenizer to a batch of text examples."""
    # `padding="max_length"` ensures all sequences have the same length (512 tokens)
    # `truncation=True` cuts off sequences longer than max_length
    # `max_length=512` is a common choice for BERT-family models
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Apply the tokenization function to all splits in the dataset (`train` and `test`)
# `batched=True` processes multiple examples at once for speed
print("Tokenizing datasets (this may take a moment)...")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# Prepare the dataset for the Trainer:
# Remove columns not needed by the model (original text, pandas index)
# Rename the 'label' column to 'labels' (expected by Trainer)
# Set the format to PyTorch tensors
tokenized_datasets = tokenized_datasets.remove_columns(["text", "__index_level_0__"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

print("\nTokenized dataset structure (ready for training):")
print(tokenized_datasets)

# Assign the final processed datasets to variables for clarity
final_train_dataset = tokenized_datasets["train"]
final_eval_dataset = tokenized_datasets["test"]

# ==============================================================================
# Model Loading
# ==============================================================================
# Load the pre-trained DistilBERT model, configured for sequence classification
# `num_labels=2` specifies that we have two output classes (positive/negative)
print(f"\nLoading pre-trained model: {MODEL_NAME} for Sequence Classification...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Move the model to the appropriate device (GPU if available, otherwise CPU)
model.to(device)
print(f"Model loaded and moved to {device}.")

# ==============================================================================
# Evaluation Metric Definition
# ==============================================================================
# Load standard metrics using Hugging Face's 'evaluate' library
print("\nLoading evaluation metrics (accuracy, f1, precision, recall)...")
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

# Define a function that the Trainer will call to compute metrics during evaluation
def compute_metrics(eval_pred):
    """Computes accuracy, F1, precision, and recall from model predictions."""
    # `eval_pred` is a tuple containing model predictions (logits) and true labels
    logits, labels = eval_pred
    # Convert logits to predictions by taking the index of the highest probability (argmax)
    predictions = np.argmax(logits, axis=-1)

    # Calculate metrics using the loaded evaluate functions
    # 'binary' averaging is suitable for binary classification, focusing on the positive class (label=1)
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="binary")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="binary")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="binary")["f1"]

    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# ==============================================================================
# Training Configuration (TrainingArguments)
# ==============================================================================
# Configure the training process using the TrainingArguments class
print("\nSetting up Training Arguments...")
training_args = TrainingArguments(
    # --- Core Parameters ---
    output_dir=DISTILBERT_TRAINING_OUTPUT_DIR, # Directory to save checkpoints and logs
    num_train_epochs=NUM_EPOCHS,               # Total number of training epochs
    learning_rate=LEARNING_RATE,               # Optimizer learning rate

    # --- Batch Sizes ---
    per_device_train_batch_size=TRAIN_BATCH_SIZE, # Batch size for training
    per_device_eval_batch_size=EVAL_BATCH_SIZE,   # Batch size for evaluation

    # --- Regularization ---
    weight_decay=WEIGHT_DECAY,                 # Weight decay for regularization

    # --- Evaluation and Saving Strategy ---
    # Use 'eval_strategy' if 'evaluation_strategy' caused errors previously
    # evaluation_strategy="epoch",      # Evaluate performance at the end of each epoch
    eval_strategy="epoch",          # Alternative name for older library versions
    save_strategy="epoch",            # Save a model checkpoint at the end of each epoch
    load_best_model_at_end=True,      # Reload the best model (based on metric) after training
    metric_for_best_model="f1",       # Metric used to identify the 'best' model (e.g., highest F1)
    greater_is_better=True,           # Indicates that a higher value for the metric is better

    # --- Logging and Reporting ---
    logging_strategy="epoch",         # Log metrics at the end of each epoch
    report_to="none",                 # Disable integration with tools like Wandb/TensorBoard

    # --- Other Options ---
    push_to_hub=False,                # Set to True to upload model to Hugging Face Hub
    # fp16=torch.cuda.is_available(), # Uncomment to enable mixed-precision training (faster on compatible GPUs)
)

# ==============================================================================
# Trainer Initialization
# ==============================================================================
# Initialize the Trainer, which orchestrates the fine-tuning process
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,                         # The model to be fine-tuned
    args=training_args,                  # Training configuration
    train_dataset=final_train_dataset,   # Training data (tokenized)
    eval_dataset=final_eval_dataset,     # Evaluation data (tokenized)
    tokenizer=tokenizer,                 # Tokenizer used for preprocessing
    compute_metrics=compute_metrics,     # Function to calculate evaluation metrics
)

# ==============================================================================
# Model Fine-Tuning
# ==============================================================================
# Start the fine-tuning process
print("\nStarting fine-tuning...")
try:
    train_result = trainer.train()
    print("Fine-tuning finished.")
    # Log some training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit() # Exit if training fails

# ==============================================================================
# Final Evaluation
# ==============================================================================
# Evaluate the best model (loaded automatically if load_best_model_at_end=True)
# on the evaluation dataset (which is the sampled test set in this configuration)
print("\nEvaluating the final fine-tuned model on the evaluation set...")
try:
    eval_results = trainer.evaluate(eval_dataset=final_eval_dataset)
    print("\n--- Final Evaluation Results ---")
    # Print metrics in a user-friendly format
    for key, value in eval_results.items():
        metric_name = key.replace("eval_", "").capitalize()
        print(f"{metric_name}: {value:.4f}")
    # Save evaluation metrics
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results) # Saves metrics.json in output_dir
except Exception as e:
    print(f"An error occurred during evaluation: {e}")

# ==============================================================================
# Save Final Model and Results
# ==============================================================================
# --- Save Model & Tokenizer ---
# Save the best performing model and its tokenizer to the designated directory
print(f"\nSaving the best fine-tuned model and tokenizer to {DISTILBERT_MODEL_SAVE_DIR}...")
try:
    trainer.save_model(DISTILBERT_MODEL_SAVE_DIR)
    # Tokenizer is typically saved automatically by save_model in recent versions
    # tokenizer.save_pretrained(DISTILBERT_MODEL_SAVE_DIR) # Explicit save if needed
    print("Model and tokenizer saved successfully.")
except Exception as e:
    print(f"Error saving final model/tokenizer: {e}")

# --- Save Evaluation Report & Confusion Matrix ---
print(f"\nSaving detailed evaluation report and confusion matrix to {DISTILBERT_RESULTS_SUBDIR}...")

# Save detailed metrics report to a text file
metrics_report_file = os.path.join(DISTILBERT_RESULTS_SUBDIR, 'distilbert_fine_tuned_metrics_report.txt')
try:
    with open(metrics_report_file, 'w') as f:
        f.write(f"--- DistilBERT ({MODEL_NAME}) Fine-Tuned Evaluation Report ---\n\n")
        f.write(f"Model Source: {MODEL_NAME}\n")
        f.write(f"Training Epochs Configured: {NUM_EPOCHS}\n")
        # Include actual epochs trained if available from train_result
        if 'train_result' in locals() and hasattr(train_result, 'metrics'):
             f.write(f"Training Epochs Completed: {train_result.metrics.get('epoch', 'N/A')}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Train Batch Size: {TRAIN_BATCH_SIZE}\n")
        f.write(f"Eval Batch Size: {EVAL_BATCH_SIZE}\n")
        f.write(f"Weight Decay: {WEIGHT_DECAY}\n")
        f.write(f"Training Sample Size: {len(final_train_dataset)}\n")
        f.write(f"Evaluation Sample Size: {len(final_eval_dataset)}\n\n")
        f.write(f"--- Performance on Evaluation Set ({len(final_eval_dataset)} samples) ---\n")
        # Write metrics from the eval_results dictionary
        if 'eval_results' in locals():
            for key, value in eval_results.items():
                 metric_name = key.replace("eval_", "").capitalize()
                 f.write(f"{metric_name}: {value:.4f}\n")
        else:
            f.write("Evaluation results dictionary not found.\n")

    print(f"Metrics report saved to {metrics_report_file}")
except Exception as e:
    print(f"Error saving metrics report file: {e}")

# Generate and save confusion matrix plot
print("Generating confusion matrix for the evaluation set...")
try:
    # Get predictions on the evaluation dataset used during training
    predictions_output = trainer.predict(final_eval_dataset)
    y_true = predictions_output.label_ids         # True labels
    y_pred = np.argmax(predictions_output.predictions, axis=1) # Predicted labels

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Define path for saving the plot
    cm_plot_file = os.path.join(DISTILBERT_RESULTS_SUBDIR, 'distilbert_fine_tuned_confusion_matrix.png')

    # Create the plot using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - Fine-Tuned DistilBERT ({MODEL_NAME})\n(Eval Set Size: {len(final_eval_dataset)})')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout() # Adjust layout to prevent labels overlapping
    plt.savefig(cm_plot_file) # Save the plot to the file
    print(f"Confusion matrix plot saved to {cm_plot_file}")
    plt.close() # Close the plot figure to free up memory
except Exception as e:
    print(f"Error generating or saving confusion matrix: {e}")


print("\n--- DistilBERT fine-tuning script finished successfully. ---")