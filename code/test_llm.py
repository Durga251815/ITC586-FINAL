# ==============================================================================
# Import Necessary Libraries
# ==============================================================================
import os                       # For operating system functions (paths)
import re                       # For regular expressions (text cleaning)
import random                   # For selecting random reviews
import torch                    # PyTorch library
from transformers import (
    AutoTokenizer,                  # Loads the correct tokenizer
    AutoModelForSequenceClassification  # Loads the sequence classification model
)
import numpy as np              # For argmax

print("--- Fine-Tuned DistilBERT - Interactive Testing Script ---")

# ==============================================================================
# Configuration
# ==============================================================================
# --- File Paths (Assumes running from project/ directory) ---
DATA_DIR = os.path.join('data', 'aclImdb')
TEST_DIR = os.path.join(DATA_DIR, 'test') # Path to the original test data
# --- !! IMPORTANT: Set this to the directory where your fine-tuned model was saved !! ---
MODEL_SAVE_DIR = os.path.join('models', 'distilbert_fine_tuned')

# ==============================================================================
# Setup: GPU Check and Model/Tokenizer Loading
# ==============================================================================
# --- Check for GPU availability ---
if torch.cuda.is_available():
    print("\nGPU detected. Using CUDA.")
    device = torch.device("cuda")
else:
    print("\nNo GPU detected. Using CPU.")
    device = torch.device("cpu")

# --- Load Fine-Tuned Model and Tokenizer ---
print(f"\nLoading fine-tuned model and tokenizer from: {MODEL_SAVE_DIR}...")
try:
    # Load the tokenizer associated with the fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SAVE_DIR)
    # Load the fine-tuned sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_DIR)
    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()
    print("Fine-tuned model and tokenizer loaded successfully.")
except OSError:
    print(f"Error: Model or tokenizer not found in directory: {MODEL_SAVE_DIR}")
    print("Please ensure you have run the fine-tuning script and the model was saved correctly.")
    exit()
except Exception as e:
    print(f"An error occurred loading the model/tokenizer: {e}")
    exit()

# ==============================================================================
# Helper Functions
# ==============================================================================

# --- Text Cleaning Function (Apply ONLY if used BEFORE tokenization during fine-tuning) ---
# If your fine-tuning script cleaned HTML *before* tokenizing, keep this. Otherwise, remove it.
def clean_text(text):
    """Basic cleaning: Removes HTML tags."""
    # Remove HTML tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # NOTE: The tokenizer might handle lowercasing and other normalization.
    # Avoid doing too much cleaning here unless it perfectly matches the fine-tuning preprocessing.
    return text

# --- Function to predict sentiment using the loaded DistilBERT model ---
def predict_sentiment_distilbert(text):
    """Cleans (if necessary), tokenizes, and predicts sentiment using the fine-tuned model."""
    # 1. Clean text (only if the same cleaning was done before tokenization in fine-tuning)
    cleaned_text = clean_text(text)

    # 2. Tokenize the input text
    # `return_tensors='pt'` ensures the output is PyTorch tensors
    # `truncation=True`, `padding=True` should ideally match fine-tuning settings if possible,
    # but for single inference, often just `truncation=True` and letting the model handle padding is okay.
    # Max length should ideally match training but 512 is standard.
    inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # 3. Move tokenized inputs to the same device as the model (GPU/CPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 4. Perform inference without calculating gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # 5. Get the raw prediction scores (logits)
    logits = outputs.logits

    # 6. Calculate probabilities using softmax
    probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0] # Get probabilities for the first (only) input

    # 7. Get the predicted class index (0 or 1) by finding the index with the highest logit/probability
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    # 8. Map class index to sentiment label
    sentiment = "Positive" if predicted_class_id == 1 else "Negative"

    # 9. Get the confidence score (probability of the predicted class)
    confidence = probabilities[predicted_class_id]

    return sentiment, confidence

# --- Function to load test filenames (same as baseline test script) ---
def load_test_filenames(test_data_dir):
    """Loads lists of positive and negative review filenames from the test set."""
    pos_files = []
    neg_files = []
    try:
        pos_dir = os.path.join(test_data_dir, 'pos')
        neg_dir = os.path.join(test_data_dir, 'neg')
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith(".txt")]
        neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith(".txt")]
    except FileNotFoundError:
        print(f"Error: Test data directory not found at {test_data_dir}")
        return None, None
    return pos_files, neg_files

# ==============================================================================
# Load Test Data for Random Selection
# ==============================================================================
print("\nLoading test data filenames for random selection option...")
pos_review_files, neg_review_files = load_test_filenames(TEST_DIR)
if pos_review_files is None or neg_review_files is None or not pos_review_files or not neg_review_files:
    print("Could not load test filenames. Random selection from test set disabled.")
    allow_random_selection = False
else:
    print(f"Found {len(pos_review_files)} positive and {len(neg_review_files)} negative test reviews.")
    allow_random_selection = True

# ==============================================================================
# Main Interaction Loop
# ==============================================================================
while True:
    # --- Display Options ---
    print("\n--- Options ---")
    print("1: Enter your own review to predict sentiment")
    if allow_random_selection:
        print("2: Predict sentiment for a random POSITIVE review from the test set")
        print("3: Predict sentiment for a random NEGATIVE review from the test set")
    print("0: Exit")

    # --- Get User Choice ---
    choice = input("Enter your choice: ")

    # --- Process Choice ---
    if choice == '1':
        # Option 1: User enters a review
        user_review = input("\nPlease enter your movie review:\n")
        if user_review.strip():
            # Predict sentiment using the fine-tuned DistilBERT model
            pred_sentiment, pred_confidence = predict_sentiment_distilbert(user_review)
            print(f"\nPredicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.4f})")
        else:
            print("No review entered.")

    elif choice == '2' and allow_random_selection:
        # Option 2: Random Positive Review from Test Set
        random_file = random.choice(pos_review_files)
        actual_sentiment = "Positive" # We know this because we picked from the 'pos' folder
        try:
            # Read the review text from the randomly selected file
            with open(random_file, 'r', encoding='utf-8') as f:
                review_text = f.read()

            # Display information and prediction
            print(f"\n--- Random Positive Review (from {os.path.basename(random_file)}) ---")
            print(f"Review Text:\n{review_text[:500]}...") # Show first 500 chars
            print("-" * 20)
            print(f"Actual Sentiment:    {actual_sentiment}")
            pred_sentiment, pred_confidence = predict_sentiment_distilbert(review_text)
            print(f"Predicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.4f})")

            # Compare prediction to actual known sentiment
            if pred_sentiment == actual_sentiment:
                print("Result: Prediction CORRECT!")
            else:
                print("Result: Prediction WRONG.")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading or processing file {random_file}: {e}")

    elif choice == '3' and allow_random_selection:
        # Option 3: Random Negative Review from Test Set
        random_file = random.choice(neg_review_files)
        actual_sentiment = "Negative" # We know this because we picked from the 'neg' folder
        try:
            # Read the review text
            with open(random_file, 'r', encoding='utf-8') as f:
                review_text = f.read()

            # Display information and prediction
            print(f"\n--- Random Negative Review (from {os.path.basename(random_file)}) ---")
            print(f"Review Text:\n{review_text[:500]}...") # Show first 500 chars
            print("-" * 20)
            print(f"Actual Sentiment:    {actual_sentiment}")
            pred_sentiment, pred_confidence = predict_sentiment_distilbert(review_text)
            print(f"Predicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.4f})")

            # Compare prediction to actual known sentiment
            if pred_sentiment == actual_sentiment:
                print("Result: Prediction CORRECT!")
            else:
                print("Result: Prediction WRONG.")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading or processing file {random_file}: {e}")

    elif choice == '0':
        # Option 0: Exit the script
        print("\nExiting script.")
        break

    else:
        # Handle invalid input
        print("\nInvalid choice. Please try again.")