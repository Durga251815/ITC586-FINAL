import os
import re
import joblib
import random

print("--- Naive Bayes Model - Interactive Testing Script ---")

# --- Configuration ---
# Assumes the script is run from the main project/ directory
DATA_DIR = os.path.join('data', 'aclImdb')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODELS_DIR = os.path.join('models')
BASELINE_MODEL_SUBDIR = os.path.join(MODELS_DIR, 'baselinemodel')

MODEL_FILE = os.path.join(BASELINE_MODEL_SUBDIR, 'naive_bayes_model.joblib')
VECTORIZER_FILE = os.path.join(BASELINE_MODEL_SUBDIR, 'tfidf_vectorizer.joblib')

# --- Text Cleaning Function (Must match the one used for training) ---
def clean_text(text):
    """Removes HTML tags and non-alphanumeric characters."""
    # Remove HTML tags
    text = re.sub(re.compile('<.*?>'), ' ', text)
    # Optional: Remove non-alphanumeric if done during training
    # text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# --- Load Model and Vectorizer ---
print("\nLoading saved model and vectorizer...")
try:
    loaded_vectorizer = joblib.load(VECTORIZER_FILE)
    loaded_model = joblib.load(MODEL_FILE)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model or vectorizer not found.")
    print(f"Please ensure '{MODEL_FILE}' and '{VECTORIZER_FILE}' exist.")
    exit()
except Exception as e:
    print(f"An error occurred loading files: {e}")
    exit()

# --- Function to predict sentiment ---
def predict_sentiment(text):
    """Cleans, transforms, and predicts sentiment for a given text."""
    cleaned = clean_text(text)
    transformed = loaded_vectorizer.transform([cleaned])
    prediction = loaded_model.predict(transformed)
    probability = loaded_model.predict_proba(transformed) # Get probabilities
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    # Get confidence score for the predicted class
    confidence = probability[0][prediction[0]]
    return sentiment, confidence

# --- Function to load test filenames ---
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

# --- Load test filenames for random selection ---
print("Loading test data filenames for random selection option...")
pos_review_files, neg_review_files = load_test_filenames(TEST_DIR)
if pos_review_files is None or neg_review_files is None or not pos_review_files or not neg_review_files:
    print("Could not load test filenames. Random selection from test set disabled.")
    allow_random_selection = False
else:
    print(f"Found {len(pos_review_files)} positive and {len(neg_review_files)} negative test reviews.")
    allow_random_selection = True

# --- Main Interaction Loop ---
while True:
    print("\n--- Options ---")
    print("1: Enter your own review to predict sentiment")
    if allow_random_selection:
        print("2: Predict sentiment for a random POSITIVE review from the test set")
        print("3: Predict sentiment for a random NEGATIVE review from the test set")
    print("0: Exit")

    choice = input("Enter your choice: ")

    if choice == '1':
        user_review = input("\nPlease enter your movie review:\n")
        if user_review.strip():
            pred_sentiment, pred_confidence = predict_sentiment(user_review)
            print(f"\nPredicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.2f})")
        else:
            print("No review entered.")

    elif choice == '2' and allow_random_selection:
        # Random Positive Review
        random_file = random.choice(pos_review_files)
        actual_sentiment = "Positive"
        try:
            with open(random_file, 'r', encoding='utf-8') as f:
                review_text = f.read()
            print(f"\n--- Random Positive Review (from {os.path.basename(random_file)}) ---")
            print(f"Review Text:\n{review_text[:500]}...") # Print first 500 chars
            print("-" * 20)
            print(f"Actual Sentiment:    {actual_sentiment}")
            pred_sentiment, pred_confidence = predict_sentiment(review_text)
            print(f"Predicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.2f})")
            if pred_sentiment == actual_sentiment:
                print("Result: Prediction CORRECT!")
            else:
                print("Result: Prediction WRONG.")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading or processing file {random_file}: {e}")

    elif choice == '3' and allow_random_selection:
        # Random Negative Review
        random_file = random.choice(neg_review_files)
        actual_sentiment = "Negative"
        try:
            with open(random_file, 'r', encoding='utf-8') as f:
                review_text = f.read()
            print(f"\n--- Random Negative Review (from {os.path.basename(random_file)}) ---")
            print(f"Review Text:\n{review_text[:500]}...") # Print first 500 chars
            print("-" * 20)
            print(f"Actual Sentiment:    {actual_sentiment}")
            pred_sentiment, pred_confidence = predict_sentiment(review_text)
            print(f"Predicted Sentiment: {pred_sentiment} (Confidence: {pred_confidence:.2f})")
            if pred_sentiment == actual_sentiment:
                print("Result: Prediction CORRECT!")
            else:
                print("Result: Prediction WRONG.")
            print("-" * 20)
        except Exception as e:
            print(f"Error reading or processing file {random_file}: {e}")

    elif choice == '0':
        print("\nExiting script.")
        break

    else:
        print("\nInvalid choice. Please try again.")