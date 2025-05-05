# Import necessary libraries
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib # For saving model and vectorizer

print("--- Naive Bayes Baseline Script ---")

# --- Configuration ---
# Assumes the script is run from the main project/ directory
DATA_DIR = os.path.join('data', 'aclImdb')
RESULTS_DIR = os.path.join('results')
MODELS_DIR = os.path.join('models')
BASELINE_RESULTS_SUBDIR = os.path.join(RESULTS_DIR, 'baselinemodel') # Specific subdir for results
BASELINE_MODEL_SUBDIR = os.path.join(MODELS_DIR, 'baselinemodel')    # Specific subdir for model files
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# --- Ensure results and models directories (including subdirs) exist ---
os.makedirs(BASELINE_RESULTS_SUBDIR, exist_ok=True)
os.makedirs(BASELINE_MODEL_SUBDIR, exist_ok=True)
print(f"Model files will be saved in: {BASELINE_MODEL_SUBDIR}")
print(f"Result files will be saved in: {BASELINE_RESULTS_SUBDIR}")

# --- Function to load data from folders ---
def load_imdb_data(data_dir):
    """Loads IMDB data from specified train/test directory."""
    texts = []
    labels = []
    # Load positive reviews
    pos_dir = os.path.join(data_dir, 'pos')
    for filename in os.listdir(pos_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(pos_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(1) # 1 for positive
            except Exception as e:
                print(f"Warning: Could not read file {filepath}: {e}")

    # Load negative reviews
    neg_dir = os.path.join(data_dir, 'neg')
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(neg_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                labels.append(0) # 0 for negative
            except Exception as e:
                print(f"Warning: Could not read file {filepath}: {e}")

    return texts, labels

# --- Text Cleaning Function ---
def clean_text(text):
    """Removes HTML tags and non-alphanumeric characters."""
    # Remove HTML tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # Remove non-alphanumeric characters (keep spaces)
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text) # Optional: Keep if you want only letters/numbers
    # Convert to lowercase
    text = text.lower()
    return text

# --- Load Data ---
print("\nLoading training data...")
train_texts, train_labels = load_imdb_data(TRAIN_DIR)
print(f"Loaded {len(train_texts)} training reviews.")

print("Loading test data...")
test_texts, test_labels = load_imdb_data(TEST_DIR)
print(f"Loaded {len(test_texts)} test reviews.")

# Check if data loading was successful
if not train_texts or not test_texts:
    print("\nError: Data loading failed. Check the DATA_DIR path and dataset structure.")
    print(f"Expected data in: {DATA_DIR}")
    exit() # Exit if data is missing

# Convert to Pandas DataFrame for easier handling (optional but good practice)
df_train = pd.DataFrame({'review': train_texts, 'sentiment': train_labels})
df_test = pd.DataFrame({'review': test_texts, 'sentiment': test_labels})

# --- Preprocessing and Feature Extraction (TF-IDF) ---
print("\nApplying text cleaning...")
df_train['review_cleaned'] = df_train['review'].apply(clean_text)
df_test['review_cleaned'] = df_test['review'].apply(clean_text)

print("Performing TF-IDF vectorization...")
# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, # Limit vocabulary size
                                   stop_words='english',
                                   ngram_range=(1, 1)) # Use unigrams

# Fit on training data and transform both training and test data
X_train_tfidf = tfidf_vectorizer.fit_transform(df_train['review_cleaned'])
X_test_tfidf = tfidf_vectorizer.transform(df_test['review_cleaned'])

print(f"Shape of TF-IDF training matrix: {X_train_tfidf.shape}")
print(f"Shape of TF-IDF test matrix: {X_test_tfidf.shape}")

# Get the target variables
y_train = df_train['sentiment']
y_test = df_test['sentiment']

# --- Train Naive Bayes Model ---
print("\nTraining Naive Bayes model...")
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
print("Model training complete.")

# --- Save the Trained Model and Vectorizer ---
print(f"\nSaving model and vectorizer to {BASELINE_MODEL_SUBDIR}...")

# Define file paths within the subdirectory
model_file = os.path.join(BASELINE_MODEL_SUBDIR, 'naive_bayes_model.joblib')
vectorizer_file = os.path.join(BASELINE_MODEL_SUBDIR, 'tfidf_vectorizer.joblib')

# Save the objects
try:
    joblib.dump(nb_model, model_file)
    joblib.dump(tfidf_vectorizer, vectorizer_file)
    print(f"Model saved to {model_file}")
    print(f"Vectorizer saved to {vectorizer_file}")
except Exception as e:
    print(f"Error saving model/vectorizer: {e}")


# --- Evaluate Model ---
print("\nEvaluating model on test data...")
y_pred = nb_model.predict(X_test_tfidf)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
# Calculate precision, recall, f1 for the positive class (label=1) specifically if needed
precision_pos, recall_pos, f1_pos, _ = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[1])
cm = confusion_matrix(y_test, y_pred)

print("\n--- Evaluation Results ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision (Positive Class): {precision_pos[0]:.4f}") # Access the first element
print(f"Recall (Positive Class): {recall_pos[0]:.4f}")     # Access the first element
print(f"F1-Score (Positive Class): {f1_pos[0]:.4f}")      # Access the first element
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(cm)

# --- Save Evaluation Results ---
print(f"\nSaving evaluation results to {BASELINE_RESULTS_SUBDIR}...")

# Save metrics to a text file
metrics_file = os.path.join(BASELINE_RESULTS_SUBDIR, 'naive_bayes_metrics.txt')
try:
    with open(metrics_file, 'w') as f:
        f.write("--- Naive Bayes Baseline Evaluation Results ---\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (Positive Class): {precision_pos[0]:.4f}\n")
        f.write(f"Recall (Positive Class): {recall_pos[0]:.4f}\n")
        f.write(f"F1-Score (Positive Class): {f1_pos[0]:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(np.array2string(cm))
    print(f"Metrics saved to {metrics_file}")
except Exception as e:
    print(f"Error saving metrics file: {e}")


# Save confusion matrix plot
cm_plot_file = os.path.join(BASELINE_RESULTS_SUBDIR, 'naive_bayes_confusion_matrix.png')
try:
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Naive Bayes Baseline')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout() # Adjust layout
    plt.savefig(cm_plot_file)
    print(f"Confusion matrix plot saved to {cm_plot_file}")
    # plt.show() # Uncomment to display the plot directly
    plt.close() # Close plot to free memory
except Exception as e:
    print(f"Error saving confusion matrix plot: {e}")

print("\n--- Baseline model script finished. ---")