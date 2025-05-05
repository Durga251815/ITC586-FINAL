# ==============================================================================
# Import Necessary Libraries
# ==============================================================================
import streamlit as st          # For creating the web dashboard
import os                       # For interacting with the operating system (paths)
import re                       # For regular expressions (text cleaning)
import random                   # For selecting random reviews
import joblib                   # For loading the Naive Bayes model/vectorizer
import torch                    # PyTorch library
from transformers import (
    AutoTokenizer,                  # Loads the correct tokenizer for DistilBERT
    AutoModelForSequenceClassification  # Loads the DistilBERT model
)
import numpy as np              # For numerical operations (argmax, softmax)
import pandas as pd             # Potentially useful for displaying data/metrics
from PIL import Image           # For displaying images (like confusion matrices)

# ==============================================================================
# Page Configuration - MUST BE THE FIRST STREAMLIT COMMAND
# ==============================================================================
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# Configuration and Paths (Define paths AFTER page config)
# ==============================================================================
# --- Assume script is run from the main project/ directory ---
DATA_DIR = os.path.join('data', 'aclImdb')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODELS_DIR = os.path.join('models')
RESULTS_DIR = os.path.join('results')

# --- Paths for Baseline Model ---
BASELINE_MODEL_DIR = os.path.join(MODELS_DIR, 'baselinemodel')
BASELINE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'baselinemodel')
NB_MODEL_FILE = os.path.join(BASELINE_MODEL_DIR, 'naive_bayes_model.joblib')
NB_VECTORIZER_FILE = os.path.join(BASELINE_MODEL_DIR, 'tfidf_vectorizer.joblib')
NB_METRICS_FILE = os.path.join(BASELINE_RESULTS_DIR, 'naive_bayes_metrics.txt')
NB_CM_FILE = os.path.join(BASELINE_RESULTS_DIR, 'naive_bayes_confusion_matrix.png')

# --- Paths for DistilBERT Model ---
DISTILBERT_MODEL_DIR = os.path.join(MODELS_DIR, 'distilbert_fine_tuned')
DISTILBERT_RESULTS_DIR = os.path.join(RESULTS_DIR, 'distilbert')
DISTILBERT_METRICS_FILE = os.path.join(DISTILBERT_RESULTS_DIR, 'distilbert_fine_tuned_metrics_report.txt')
DISTILBERT_CM_FILE = os.path.join(DISTILBERT_RESULTS_DIR, 'distilbert_fine_tuned_confusion_matrix.png')

# --- Path for EDA plots ---
EDA_INFO_DIR = os.path.join(RESULTS_DIR, 'dataset_info')
REVIEW_LENGTH_PLOT = os.path.join(EDA_INFO_DIR, 'review_length_distribution.png')
COMMON_WORDS_PLOT = os.path.join(EDA_INFO_DIR, 'most_common_words.png')


# ==============================================================================
# Caching: Load Models and Data Once (Define functions AFTER page config)
# ==============================================================================
@st.cache_resource
def load_naive_bayes_model():
    """Loads the Naive Bayes model and vectorizer."""
    print("Attempting to load Naive Bayes model and vectorizer...")
    try:
        model = joblib.load(NB_MODEL_FILE)
        vectorizer = joblib.load(NB_VECTORIZER_FILE)
        print("Naive Bayes model and vectorizer loaded successfully.")
        return model, vectorizer
    except FileNotFoundError:
        print(f"ERROR: Naive Bayes model or vectorizer not found in {BASELINE_MODEL_DIR}")
        return None, None
    except Exception as e:
        print(f"ERROR loading Naive Bayes components: {e}")
        return None, None

@st.cache_resource
def load_distilbert_model():
    """Loads the fine-tuned DistilBERT model and tokenizer."""
    print("Attempting to load DistilBERT model and tokenizer...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DistilBERT model onto device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_MODEL_DIR)
        model.to(device)
        model.eval()
        print("DistilBERT model and tokenizer loaded successfully.")
        return model, tokenizer, device
    except OSError:
        print(f"ERROR: DistilBERT fine-tuned model or tokenizer not found in {DISTILBERT_MODEL_DIR}")
        return None, None, None
    except Exception as e:
        print(f"ERROR loading DistilBERT components: {e}")
        return None, None, None

@st.cache_data
def load_test_filenames(test_data_dir):
    """Loads lists of positive and negative review filenames."""
    print("Attempting to load test filenames...")
    pos_files = []
    neg_files = []
    error_msg = None
    allow_random = False
    try:
        pos_dir = os.path.join(test_data_dir, 'pos')
        neg_dir = os.path.join(test_data_dir, 'neg')
        if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
             raise FileNotFoundError(f"'pos' or 'neg' dir not found in {test_data_dir}")
        pos_files = [os.path.join(pos_dir, f) for f in os.listdir(pos_dir) if f.endswith(".txt")]
        neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.endswith(".txt")]
        if not pos_files or not neg_files:
            error_msg = f"No review .txt files found in {pos_dir} or {neg_dir}."
        else:
            print(f"Loaded {len(pos_files)} positive and {len(neg_files)} negative test filenames.")
            allow_random = True
    except FileNotFoundError as e:
        error_msg = f"Error: Test data directory not found: {e}"
    except Exception as e:
        error_msg = f"Error loading test filenames: {e}"
    if error_msg:
         print(f"WARNING: {error_msg}")
    return pos_files, neg_files, allow_random, error_msg


# ==============================================================================
# Helper Functions (Define AFTER page config)
# ==============================================================================
def clean_text_naive_bayes(text):
    text = re.sub(re.compile('<.*?>'), ' ', text)
    return text

def clean_text_distilbert(text):
    text = re.sub(re.compile('<.*?>'), ' ', text)
    return text

def predict_naive_bayes(text, model, vectorizer):
    if model is None or vectorizer is None:
        st.error("Naive Bayes model/vectorizer not available for prediction.")
        return "Error", 0.0
    try:
        cleaned_text = clean_text_naive_bayes(text)
        transformed_text = vectorizer.transform([cleaned_text])
        prediction = model.predict(transformed_text)[0]
        probability = model.predict_proba(transformed_text)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = probability[prediction]
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error during Naive Bayes prediction: {e}")
        return "Error", 0.0

def predict_distilbert(text, model, tokenizer, device):
    if model is None or tokenizer is None:
        st.error("DistilBERT model/tokenizer not available for prediction.")
        return "Error", 0.0
    try:
        cleaned_text = clean_text_distilbert(text)
        inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        sentiment = "Positive" if predicted_class_id == 1 else "Negative"
        confidence = probabilities[predicted_class_id]
        return sentiment, confidence
    except Exception as e:
        st.error(f"Error during DistilBERT prediction: {e}")
        return "Error", 0.0

# NOTE: display_metrics and display_image are now used in the main area, not just sidebar
def display_metrics(metrics_file):
    """Reads metrics from a text file and displays them."""
    try:
        with open(metrics_file, 'r') as f:
            st.text(f.read()) # Use st.text for better formatting control if needed
    except FileNotFoundError:
        st.warning(f"Metrics file not found: {metrics_file}")
    except Exception as e:
        st.error(f"Error reading metrics file: {e}")

def display_image(image_path, caption=""):
    """Loads and displays an image."""
    try:
        image = Image.open(image_path)
        st.image(image, caption=caption, use_container_width ='auto')
    except FileNotFoundError:
        st.warning(f"Image file not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image {image_path}: {e}")


# ==============================================================================
# Load resources using caching (Call cached functions AFTER definitions)
# ==============================================================================
nb_model, nb_vectorizer = load_naive_bayes_model()
distilbert_model, distilbert_tokenizer, device = load_distilbert_model()
pos_review_files, neg_review_files, test_files_loaded, test_load_error = load_test_filenames(TEST_DIR)


# ==============================================================================
# Streamlit App Layout (Main UI Rendering)
# ==============================================================================

# --- Title ---
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("A dashboard demonstrating sentiment prediction using a baseline Naive Bayes model and a fine-tuned DistilBERT LLM.")

# --- Sidebar for Model Selection ONLY ---
st.sidebar.header("⚙️ Controls")

model_options = []
if nb_model is not None and nb_vectorizer is not None:
    model_options.append("Naive Bayes (Baseline)")
if distilbert_model is not None and distilbert_tokenizer is not None:
    model_options.append("DistilBERT (Fine-tuned LLM)")

if not model_options:
     st.sidebar.error("No models could be loaded. Please check file paths and logs.")
     st.error("Critical Error: No models loaded. Cannot proceed.")
     st.stop()

default_model_index = 0
if "DistilBERT (Fine-tuned LLM)" in model_options:
    default_model_index = model_options.index("DistilBERT (Fine-tuned LLM)")

model_choice = st.sidebar.selectbox(
    "Choose a Model:",
    model_options,
    index=default_model_index,
    key="model_selector"
)
# --- End Sidebar ---


# ======================== SECTION ORDERING START ==============================

# --- Section 1: Main Area for Interaction ---
st.header(f"Test the {model_choice} Model")
tab1, tab2 = st.tabs(["✍️ Enter Your Own Review", "🎲 Test Random Sample"])

# --- Tab 1: Manual Input ---
with tab1:
    st.subheader("Analyze Your Review Text")
    user_text = st.text_area("Enter movie review text here:", height=150, key=f"manual_input_{model_choice}")
    predict_button = st.button("Analyze Sentiment", key=f"manual_predict_{model_choice}")

    if predict_button and user_text.strip():
        st.markdown("---")
        st.subheader("Analysis Result:")
        with st.spinner(f"Analyzing using {model_choice}..."):
            if model_choice == "Naive Bayes (Baseline)":
                sentiment, confidence = predict_naive_bayes(user_text, nb_model, nb_vectorizer)
            else: # DistilBERT
                sentiment, confidence = predict_distilbert(user_text, distilbert_model, distilbert_tokenizer, device)

            if sentiment == "Positive": st.success(f"Predicted Sentiment: **{sentiment}**")
            elif sentiment == "Negative": st.error(f"Predicted Sentiment: **{sentiment}**")
            else: st.warning(f"Prediction failed: {sentiment}")
            if sentiment != "Error":
                 st.progress(float(confidence))
                 st.markdown(f"**Confidence:** {confidence:.4f}")
    elif predict_button and not user_text.strip():
        st.warning("Please enter some text to analyze.")

# --- Tab 2: Random Sample Testing ---
with tab2:
    st.subheader("Analyze a Random Review from Test Set")

    if not test_files_loaded:
        st.error(f"Could not load test review files. Cannot perform random sampling. Error: {test_load_error}")
    else:
        col1, col2 = st.columns(2)
        with col1: pos_button = st.button("Analyze Random POSITIVE Review", key=f"random_pos_{model_choice}")
        with col2: neg_button = st.button("Analyze Random NEGATIVE Review", key=f"random_neg_{model_choice}")

        state_prefix = model_choice.replace(" ", "_").replace("(", "").replace(")", "")
        text_key = f"{state_prefix}_random_review_text"; actual_key = f"{state_prefix}_random_review_actual"; predicted_key = f"{state_prefix}_random_review_predicted"; confidence_key = f"{state_prefix}_random_review_confidence"; correct_key = f"{state_prefix}_random_review_correct"
        if text_key not in st.session_state: st.session_state[text_key] = ""; st.session_state[actual_key] = ""; st.session_state[predicted_key] = ""; st.session_state[confidence_key] = 0.0; st.session_state[correct_key] = None

        if pos_button:
            st.session_state[actual_key] = "Positive"; random_file = random.choice(pos_review_files)
            try:
                with open(random_file, 'r', encoding='utf-8') as f: st.session_state[text_key] = f.read()
                with st.spinner(f"Analyzing using {model_choice}..."):
                     if model_choice == "Naive Bayes (Baseline)": sentiment, confidence = predict_naive_bayes(st.session_state[text_key], nb_model, nb_vectorizer)
                     else: sentiment, confidence = predict_distilbert(st.session_state[text_key], distilbert_model, distilbert_tokenizer, device)
                st.session_state[predicted_key] = sentiment; st.session_state[confidence_key] = confidence; st.session_state[correct_key] = (sentiment == st.session_state[actual_key]) if sentiment != "Error" else None
            except Exception as e: st.error(f"Error processing positive review: {e}"); st.session_state[text_key] = "Error loading review."; st.session_state[correct_key] = None

        if neg_button:
            st.session_state[actual_key] = "Negative"; random_file = random.choice(neg_review_files)
            try:
                with open(random_file, 'r', encoding='utf-8') as f: st.session_state[text_key] = f.read()
                with st.spinner(f"Analyzing using {model_choice}..."):
                     if model_choice == "Naive Bayes (Baseline)": sentiment, confidence = predict_naive_bayes(st.session_state[text_key], nb_model, nb_vectorizer)
                     else: sentiment, confidence = predict_distilbert(st.session_state[text_key], distilbert_model, distilbert_tokenizer, device)
                st.session_state[predicted_key] = sentiment; st.session_state[confidence_key] = confidence; st.session_state[correct_key] = (sentiment == st.session_state[actual_key]) if sentiment != "Error" else None
            except Exception as e: st.error(f"Error processing negative review: {e}"); st.session_state[text_key] = "Error loading review."; st.session_state[correct_key] = None

        if st.session_state[text_key]:
            st.markdown("---"); st.subheader("Random Review Analysis")
            st.text_area("Review Text:", st.session_state[text_key], height=200, key=f"random_display_{model_choice}")
            st.markdown(f"**Actual Sentiment:** {st.session_state[actual_key]}"); st.markdown(f"**Predicted Sentiment:** {st.session_state[predicted_key]}")
            if st.session_state[predicted_key] != "Error":
                 st.progress(float(st.session_state[confidence_key])); st.markdown(f"**Confidence:** {st.session_state[confidence_key]:.4f}")
                 if st.session_state[correct_key] is True: st.success("✅ Prediction CORRECT!")
                 elif st.session_state[correct_key] is False: st.error("❌ Prediction WRONG.")


# --- Section 2: Dataset Insights (EDA) Section ---
st.markdown("---") # Add a separator
st.header("Dataset Insights (EDA)")
col_eda1, col_eda2 = st.columns(2)
with col_eda1:
    st.subheader("Review Length Distribution")
    display_image(REVIEW_LENGTH_PLOT, "Histogram of word counts per review")
with col_eda2:
    st.subheader("Most Common Words")
    display_image(COMMON_WORDS_PLOT, "Top words after cleaning (excluding stopwords)")


# --- Section 3: Model Performance Report Section (Moved from Sidebar) ---
st.markdown("---") # Add a separator
st.header(f"Performance Report: {model_choice}")

# Use columns for better layout in the main area as well
perf_col1, perf_col2 = st.columns(2)

with perf_col1:
    st.subheader("Evaluation Metrics")
    if model_choice == "Naive Bayes (Baseline)":
        display_metrics(NB_METRICS_FILE)
    elif model_choice == "DistilBERT (Fine-tuned LLM)":
        display_metrics(DISTILBERT_METRICS_FILE)

with perf_col2:
    st.subheader("Confusion Matrix")
    if model_choice == "Naive Bayes (Baseline)":
        display_image(NB_CM_FILE)
    elif model_choice == "DistilBERT (Fine-tuned LLM)":
        display_image(DISTILBERT_CM_FILE)
# ======================== SECTION ORDERING END ==============================


# --- Final Sidebar Info ---
st.sidebar.markdown("---")
st.sidebar.info("Dashboard created to demonstrate IMDB sentiment analysis project.")

# --- End of Script ---
print("\nStreamlit dashboard script execution finished (server may still be running).")